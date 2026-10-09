"""Phase-3 orchestration and artifacts; Phase-1/2 state is read-only."""
from dataclasses import replace

import torch

from joint_reduction.continuous_joint_geometry import fit_continuous_joint, initialise_continuous
from joint_reduction.direct_gaussian_joint import JointChart, fit_joint_from_gaussian_tracks
from joint_reduction.direct_rgbd_joint import (RGBDJointProblem, compare_candidates,
    fit_joint_from_rgbd, heldout_camera_validation, measurement_terms)
from joint_reduction.gaussian_transport_metric import canonical_moments
from joint_reduction.geometric_joint_fit import relative_trajectory
from joint_reduction.joint_information import (accumulate_gaussian_information,
    gaussian_joint_influence, information_inverse, accumulate_block_information)
from joint_reduction.joint_uncertainty import (cluster_robust_covariance,
    curvature_covariance, curvature_evidence, heldout_predictive_gain, model_adequacy)


def motion_image_information(state, cameras, observation, cfg):
    from estimator.renderer_bridge import build_linearization
    lin = build_linearization(state, cameras, eps_depth=cfg.eps_depth)
    p, r = [], []
    for ci,view in enumerate(lin.views):
        prec,resid,_e,_w = measurement_terms(view, observation[ci], cfg)
        p.append(prec);r.append(resid)
    return accumulate_gaussian_information(lin,p,r)['F_image'].cpu()


def public_fit(fit):
    keep = ('d','eta','s','c','kappa','axis','axis_point','q','length_scale',
            'energy','status','converged','history','orientation_renderer_jacobian',
            'finite_axis_fallback','curvature','uncertainty','initializer','decision')
    return {k:fit[k] for k in keep if k in fit}


def block_comparisons(rows, block_frames):
    grouped = {}
    for r in rows:
        key=(r['fold'],r['camera'],r['frame']//block_frames)
        if key not in grouped:
            grouped[key]={**r,'window':key[-1]}
            for k in ('E_P_rgb','E_R_rgb','E_P_depth','E_R_depth','gain_rgb','gain_depth','gain_total','valid_rgb','valid_depth'):
                grouped[key][k]=0.
        for k in ('E_P_rgb','E_R_rgb','E_P_depth','E_R_depth','gain_rgb','gain_depth','gain_total','valid_rgb','valid_depth'):
            grouped[key][k]+=r.get(k,0.)
    return list(grouped.values())


def discrepancy(x, charts):
    out={}
    for a,b in [('SE3','Gaussian'),('SE3','RGBD'),('Gaussian','RGBD')]:
        if charts.get(a) is None or charts.get(b) is None:continue
        total=0.
        for t in range(len(charts[a].s)):
            Ra,pa=charts[a].transform(t);Rb,pb=charts[b].transform(t)
            total+=float(((x@Ra.T+pa)-(x@Rb.T+pb)).square().sum())
        out[f'{a}_vs_{b}_rms']= (total/(len(x)*len(charts[a].s)))**.5
    return out


def fit_direct_joints(objective, rig, positions, image_information, se3_joints,
                      partition, args, writer, *, heldout=None, verbose=True):
    from estimator.rigid_part_estimator import apply_transforms
    T,N=len(positions),len(objective.mu0)
    tracks=torch.stack(positions).double()
    info=torch.stack(image_information).to(tracks.device).double()
    writer.tensor('phase1_image_information',{'F_image':info.cpu(),
        'F_prior_file':'phase1_prior_information.pt','prior_excluded':True,
        'source':'analytic rendered centre Fisher at each accepted Phase-1 state'})
    writer.tensor('phase1_positions',tracks.cpu())
    values,vectors=torch.linalg.eigh(info)
    supported=values>3*torch.finfo(info.dtype).eps*values.abs().amax(-1,keepdim=True)
    invvalues=torch.where(supported,values.clamp_min(torch.finfo(info.dtype).tiny).reciprocal(),0.)
    covariance=(vectors*invvalues[...,None,:])@vectors.transpose(-1,-2)
    writer.tensor('phase1_motion_covariance',{'image_only_pseudocovariance':covariance.cpu(),
        'rank':supported.sum(-1).cpu(),'null_directions_are_unobserved':True,
        'not_used_as_independent_joint_observations':True})
    del covariance,values,vectors,invvalues,supported
    writer.tensor('canonical_gaussians',{'mu':objective.mu0.cpu(),'R':objective.R0.cpu(),
        'colors':objective.template.colors.cpu(),'scales':objective.template.scales.cpu(),'opacity':objective.template.opacity.cpu()})
    base=[apply_transforms(objective.mu0,objective.R0,rig['labels'],rig['transforms'],t) for t in range(T)]
    estimates,selected,heat=[],[],{}
    supports, camera_rows, frame_rows, held_rows, influences, spectra, node_rows=[],[],[],[],[],[],[]
    for k in range(1,rig['transforms'].K+1):
        mask=rig['labels']==k
        if int(mask.sum())<3:
            estimates.append({'part':k,'status':'UNRESOLVED_GEOMETRY','reason':'empty or insufficient support'});selected.append(None);continue
        x=objective.mu0[mask].double()
        moments=canonical_moments(x)
        Ro,po=rig['transforms'].part_trajectory(k);Ro,po=relative_trajectory(Ro,po)
        a=fit_continuous_joint(moments['c'],Ro,po,moments['L'],max_iters=args.joint_iters,moments=moments)
        initial=JointChart(moments['c'],a['d'],a['eta'],a['s'],max(moments['L'],torch.finfo(torch.float64).tiny))
        # Preserve the exact finite-axis fallback at the chart's true singularity.
        centre_motion=torch.einsum('tij,j->ti',Ro,moments['c'])+po-moments['c']
        if float(centre_motion.norm())<=torch.finfo(torch.float64).eps*max(float(po.norm()),moments['L']):
            from estimator.manifold import so3_log
            phi=so3_log(Ro);imax=int(phi.norm(dim=1).argmax())
            if float(phi[imax].norm())>torch.finfo(torch.float64).eps:
                axis=phi[imax]/phi[imax].norm();angles=phi@axis
                initial=replace(initial,d=axis,eta=torch.zeros_like(axis),s=angles,finite_axis=True)
        a.update(initial.summary(), converged=bool(a.get('eligible')))
        gR=gP=None
        if args.joint_direct_gaussian:
            gP=fit_joint_from_gaussian_tracks(x,tracks[:,mask],info[:,mask],dt=args.dt,
                                            fixed_curvature=True,max_iters=args.joint_iters)
            gR=fit_joint_from_gaussian_tracks(x,tracks[:,mask],info[:,mask],dt=args.dt,
                                             initial=initial,max_iters=args.joint_iters)
            alternative=fit_joint_from_gaussian_tracks(x,tracks[:,mask],info[:,mask],initial=replace(gP['chart'],fixed_curvature=False),max_iters=args.joint_iters)
            if alternative['energy']<gR['energy']:gR=alternative
            grouped={}
            for row in gR['details']['blocks']:
                window=row['frame']//args.joint_block_frames
                grouped[window]=grouped.get(window,torch.zeros_like(row['score']))+row['score']
            Gu=cluster_robust_covariance(gR['information'],torch.stack(list(grouped.values())))
            Gu['source']='contiguous temporal clusters of tracks; Phase-1 cameras jointly reconstructed'
            gR['uncertainty']=Gu;gR['curvature']=curvature_covariance(gR['chart'],Gu['covariance'],gR['information'])
        problem=RGBDJointProblem(objective,mask,base_states=base,block_frames=args.joint_block_frames)
        if not args.joint_direct_rgbd:
            estimates.append({'part':k,'SE3':public_fit(a),'Gaussian':public_fit(gR) if gR else None,
                              'status':'DIAGNOSTIC_ONLY','production_authoritative':False})
            selected.append(se3_joints[k-1]);continue
        def save_weights(role):
            def save(t,ci,weights):
                import numpy as np
                path=writer.dir/'joint_robust_weights'/f'part_{k}'/role/f'frame_{t:06d}_camera_{ci:03d}.npz'
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,rgb=weights[0].cpu().numpy().astype(np.float16),
                                    depth=weights[1].cpu().numpy().astype(np.float16))
            return save
        problem.weight_writer=save_weights('P')
        cP=fit_joint_from_rgbd(problem,gP['chart'] if gP else initial,fixed_curvature=True,max_iters=args.joint_iters)
        problem.weight_writer=save_weights('R')
        cR=fit_joint_from_rgbd(problem,gR['chart'] if gR else initial,max_iters=args.joint_iters)
        alt=fit_joint_from_rgbd(problem,replace(cP['chart'],fixed_curvature=False),max_iters=args.joint_iters,diagnostics=False)
        if alt['energy']<cR['energy']:
            cR=fit_joint_from_rgbd(problem,alt['chart'],max_iters=args.joint_iters)
        train_rows=compare_candidates(problem,cP['chart'],cR['chart'])
        camera_normals={(r['frame'],r['camera']):r for r in cR['details']['camera_rows']}
        for row in train_rows:
            row['information_trace']=float(camera_normals[row['frame'],row['camera']]['A'].trace())
        validation=[];fold_fits=[]
        authoritative=heldout is not None
        if authoritative:
            hp=RGBDJointProblem(heldout,mask,base_states=base,block_frames=args.joint_block_frames)
            validation=compare_candidates(hp,cP['chart'],cR['chart'],args.camera_fold,'heldout_authoritative')
        elif args.joint_heldout_validation:
            validation,fold_fits=heldout_camera_validation(problem,cP,cR,folds=max(args.camera_folds,2),max_iters=args.joint_iters)
        blocks=block_comparisons(validation,args.joint_block_frames)
        predictive=heldout_predictive_gain(blocks,args.joint_confidence)
        adequacy=model_adequacy(blocks,args.joint_confidence)
        decision=curvature_evidence(cR['curvature'],predictive,confidence=args.joint_confidence,
            converged=cP['converged'] and cR['converged'] and rig['converged'],
            model_mismatch=adequacy['model_mismatch'],authoritative=authoritative)
        if fold_fits and not all(f['converged'] for f in fold_fits):
            decision.update(selected=None,status='UNRESOLVED_OPTIMIZATION',
                            reason='a diagnostic training fold did not converge')
        if gR:
            gp = hp if authoritative else problem
            gb = block_comparisons(compare_candidates(gp,gP['chart'],gR['chart'],
                args.camera_fold,'heldout_authoritative' if authoritative else 'training_diagnostic'),args.joint_block_frames)
            gR['decision']=curvature_evidence(gR['curvature'],heldout_predictive_gain(gb,args.joint_confidence),
                confidence=args.joint_confidence,converged=gP['converged'] and gR['converged'],authoritative=False)
            gR['decision']['validation_mode']='heldout_cross_check' if authoritative else 'training_only_cross_check'
            if not authoritative:
                gR['decision'].update(selected=None,status='DIAGNOSTIC_ONLY')
        entry={'part':k,'SE3':{**public_fit(a),'selected':(se3_joints[k-1] or {}).get('selected')},
               'Gaussian':public_fit(gR) if gR else None,'Gaussian_P':public_fit(gP) if gP else None,
               'RGBD':public_fit(cR),'RGBD_P':public_fit(cP),'decision':decision,'adequacy':adequacy,
               'disagreement':discrepancy(x,{'SE3':initial,'Gaussian':gR['chart'] if gR else None,'RGBD':cR['chart']}),
               'fast_fold_fits':[{'fold':f['fold'],'train_cameras':f['train_cameras'],
                   'heldout_cameras':f['heldout_cameras'],'P':public_fit(f['P']),'R':public_fit(f['R']),
                   'phase12_recomputed':False} for f in fold_fits]}
        # Reaggregate identical scores at a second block width; point estimate fixed.
        sensitivity={}
        for width in sorted(set([args.joint_block_frames,max(1,args.joint_block_frames//2)])):
            b=accumulate_block_information(cR['details']['camera_rows'],width)
            u=cluster_robust_covariance(cR['information'],torch.stack([v['score'] for v in b]))
            sensitivity[str(width)]={**u,'curvature':curvature_covariance(cR['chart'],u['covariance'],cR['information'])}
        entry['block_width_sensitivity']=sensitivity
        writer.tensor(f'part_{k}_joint_camera_information',cR['details']['cameras'])
        writer.tensor(f'part_{k}_joint_block_information',cR['details']['blocks'])
        entry['camera_influence']=cR['camera_influence']
        entry['information_mass_curve']=_information_mass_curve(
            x,tracks[:,mask],info[:,mask],cR,hp if authoritative else problem,args,
            heldout=authoritative)
        _plot_information_curve(writer,k,entry['information_mass_curve'])
        estimates.append(entry)
        chosen=cP if decision['selected']=='prismatic' else cR
        # No silent fallback to A/B when C cannot establish a production verdict.
        if decision['selected'] and authoritative:
            selected.append({'part':k,'selected':decision['selected'],'status':decision['status'],
                'axis':chosen['axis'].tolist(),'axis_point':chosen['axis_point'].tolist() if chosen['axis_point'] is not None else None,
                'q':chosen['q'].tolist(),'normalized_error':{'prismatic':cP['energy'],'revolute':cR['energy'],
                    'absolute_gap':cP['energy']-cR['energy'],'indistinguishable_floor':None},'direct_rgbd':True})
        else:
            selected.append({'part':k,'selected':None,'status':decision['status'],'axis':None,'axis_point':None,'q':None,
                             'normalized_error':{'prismatic':cP['energy'],'revolute':cR['energy']}})
        rows=[{**r,'part':k} for r in train_rows+validation]
        camera_rows+=rows;frame_rows+=rows;held_rows += [{**r,'part':k} for r in blocks]
        for b in cR['block_influence']:
            influences.append([k,b['camera'],b['window'],float(b['delta'].norm()),*b['delta'][:4].cpu().tolist()])
        spectrum=information_inverse(cR['information'])['spectrum']
        spectra += [[k,j,float(v)] for j,v in enumerate(spectrum)]
        support,colors=_gaussian_support(problem,cP,cR,tracks[:,mask],k)
        supports.append(support)
        for name,v in colors.items():
            if name not in heat:heat[name]=torch.zeros(N).double()
            heat[name][mask.cpu()]=v
        sid=partition.supernode_id[mask].cpu().long()
        for sn in sid.unique().tolist():
            m=sid==sn
            node_rows.append([k,sn,int(m.sum()),float(colors['support'][m].sum()),float(colors['information'][m].sum()),float(colors['leverage'][m].sum())])
        if verbose:
            print(f"  direct joint part {k}: kappa A={a['kappa']:.5g}, B={gR['kappa'] if gR else None}, C={cR['kappa']}; "
                  f"SNR={decision['curvature_snr']}, heldout gain={predictive.get('mean')}; {decision['status']}",flush=True)
    writer.json('direct_joint_estimates',{'per_part':estimates,'confidence_level':args.joint_confidence,
        'noise_model':{'rgb_sigma':objective.cfg.rgb_sigma,'depth_sigma_model':objective.cfg.depth_sigma_model,
                       'student_t_nu':objective.cfg.student_t_nu,'model_error':objective.model_error.to_dict() if objective.model_error else None},
        'fixed_appearance':True,'fixed_support':True,'orientation_renderer_jacobian':True,
        'robust_weight_storage':'joint_robust_weights/part_*/{P,R}/*.npz; float16 diagnostic weights; fitting uses full precision',
        'phase12_recomputed_for_heldout':heldout is not None})
    writer.tensor('gaussian_joint_support',supports);writer.tensor('joint_support_gaussians',supports)
    writer.csv('joint_support_supernodes',['part','supernode','n','R_over_P','information','leverage'],node_rows)
    _write_loss_tables(writer,camera_rows,frame_rows,held_rows)
    writer.csv('joint_block_influence',['part','camera','window','norm','delta_alpha1','delta_alpha2','delta_beta1','delta_beta2'],influences)
    writer.csv('joint_information_spectrum',['part','index','eigenvalue'],spectra)
    writer.json('joint_uncertainty',{'per_part':[{'part':e['part'],'decision':e.get('decision'),
        'covariance':(e.get('RGBD') or {}).get('uncertainty'),'sensitivity':e.get('block_width_sensitivity')} for e in estimates]})
    return selected,estimates,heat


def _information_mass_curve(x,tracks,information,rfit,problem,args,*,heldout=False):
    """Exact B refits at cumulative information masses; diagnostic only."""
    trace=torch.zeros(len(x)).double()
    for row in rfit['details']['gaussian']:
        trace+=torch.einsum('nij,nik,nkl->njl',row['G'],row['F_pose'],row['G']).diagonal(dim1=-2,dim2=-1).sum(-1)
    order=trace.argsort(descending=True)
    cumulative=trace[order].cumsum(0)
    rows=[]
    for fraction in (.1,.2,.3,.4,.5,.6,.7,.8,.9,1.):
        count=max(3,min(len(x),int(torch.searchsorted(cumulative,cumulative[-1]*fraction))+1))
        ids=order[:count].to(x.device)
        fit=fit_joint_from_gaussian_tracks(x[ids],tracks[:,ids],information[:,ids],
            initial=rfit['chart'],max_iters=args.joint_iters)
        comparison=compare_candidates(problem,replace(fit['chart'],fixed_curvature=True,eta=torch.zeros_like(fit['chart'].eta)),fit['chart'])
        rows.append({'information_fraction':fraction,'gaussians':count,'kappa':fit['kappa'],
            'eta':fit['eta'],'axis':fit['axis'],'q':fit['q'],'status':fit['status'],
            'rgbd_gain':sum(r['gain_total'] for r in comparison),'heldout':heldout,
            'estimator':'Gaussian tracks; full fixed support used for RGB-D evaluation'})
    return rows


def _plot_information_curve(writer,part,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    masses=[r['information_fraction'] for r in rows]
    axes[0].plot(masses,[r['kappa'] for r in rows],'o-')
    axes[0].set_ylabel('Curvature');axes[1].plot(masses,[r['rgbd_gain'] for r in rows],'o-')
    axes[1].set_ylabel('Held-out RGB-D gain' if rows[0]['heldout'] else 'Training RGB-D gain')
    for ax in axes:ax.set_xlabel('Retained information mass');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(writer.dir/f'joint_information_mass_part_{part}.png');plt.close(fig)


def _gaussian_support(problem,pfit,rfit,tracks,part):
    from estimator.manifold import so3_log
    qP,qR=pfit['chart'],rfit['chart']
    inverse=information_inverse(rfit['information'])['inverse'].cpu()
    T,N=len(qR.s),int(problem.member.sum())
    names=['alpha_mass','visible_camera_count','trace_F_xyz','joint_information','joint_leverage',
           'predicted_gain_P','predicted_gain_R','camera_score_variance','track_residual']
    out={name:torch.zeros(T,N).double() for name in names}
    out.update(gaussian_id=problem.member.nonzero().flatten().cpu(),part=part,frame=torch.arange(T),
               eig_F_xyz=torch.zeros(T,N,3).double(),condition_F_xyz=torch.zeros(T,N).double())
    cameras={};Fs=torch.zeros(T,N,3,3).double()
    p_rows={(r['frame'],r['camera']):r for r in pfit['details']['gaussian']}
    x0=problem.objective.mu0[problem.member].double()
    for row in rfit['details']['gaussian']:
        t,ci=row['frame'],row['camera'];ix=row['indices'].cpu()
        G,F,b=row['G'],row['F_pose'],row['b_pose']
        v=gaussian_joint_influence(G,F,b,inverse[ix[:,None],ix[None,:]],torch.zeros(len(ix)).double())
        out['joint_information'][t]+=v['joint_information'];out['joint_leverage'][t]+=v['leverage']
        out['alpha_mass'][t]+=row['alpha_mass'];out['visible_camera_count'][t]+=(row['alpha_mass']>0).double()
        Fs[t]+=F[:,:3,:3]
        Rp,pp=qP.transform(t);Rr,pr=qR.transform(t)
        delta=torch.cat([(x0@Rr.T+pr)-(x0@Rp.T+pp),so3_log((Rr@Rp.T)[None])[0].expand(N,3)],1).cpu()
        ref=p_rows[t,ci]
        predicted=-2*(delta*ref['b_pose']).sum(1)+torch.einsum('ni,nij,nj->n',delta,ref['F_pose'],delta)
        out['predicted_gain_R'][t]-=predicted
        cameras.setdefault(t,[]).append((delta*ref['b_pose']).sum(1))
        out['track_residual'][t]=(tracks[t]-(x0@Rr.T+pr)).norm(dim=1).cpu()
    for t in range(T):
        ev=torch.linalg.eigvalsh(Fs[t]).clamp_min(0.)
        out['eig_F_xyz'][t]=ev;out['trace_F_xyz'][t]=ev.sum(1)
        out['condition_F_xyz'][t]=torch.where(ev[:,0]>0,ev[:,-1]/ev[:,0].clamp_min(torch.finfo(torch.float64).tiny),float('inf'))
        if len(cameras.get(t,[]))>1:out['camera_score_variance'][t]=torch.stack(cameras[t]).var(0)
    out['F_image_xyz']=Fs
    out['delta_E_P']=-out['predicted_gain_P']
    out['delta_E_R']=-out['predicted_gain_R']
    out['R_support_over_P']=out['predicted_gain_R']-out['predicted_gain_P']
    scales=problem.objective.template.scales[problem.member].double().cpu().square()
    out['covariance_anisotropy']=scales.max(1).values/scales.min(1).values.clamp_min(torch.finfo(torch.float64).tiny)
    out['quadratic_reference']='P optimum; delta_E_P=0, delta_E_R includes centre and orientation terms'
    return out,{'support':out['predicted_gain_R'].sum(0),'information':out['joint_information'].sum(0),
                'leverage':out['joint_leverage'].sum(0),'visibility':out['alpha_mass'].sum(0),
                'disagreement':out['camera_score_variance'].sum(0)}


def _write_loss_tables(writer,camera_rows,frame_rows,held_rows):
    losses=['E_P_rgb','E_R_rgb','E_P_depth','E_R_depth','gain_rgb','gain_depth','gain_total']
    fields=losses+['information_trace']
    for name,groupfields,rows in [('joint_support_by_camera',['part','camera','train_or_heldout'],camera_rows),
                                ('joint_support_by_frame',['part','frame','train_or_heldout'],frame_rows)]:
        groups={}
        for r in rows:
            key=tuple(r[k] for k in groupfields)
            groups.setdefault(key,[0.]*len(fields))
            groups[key]=[a+r[f] if a is not None and r.get(f) is not None else None
                         for a,f in zip(groups[key],fields)]
        writer.csv(name,groupfields+fields,[list(k)+v for k,v in groups.items()])
    fields=['part','fold','camera','window',*losses,'valid_rgb','valid_depth']
    writer.csv('heldout_joint_comparison',fields,[[r.get(f) for f in fields] for r in held_rows])
