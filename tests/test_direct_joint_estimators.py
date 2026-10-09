"""Executable checks for the analytic direct-joint implementation contract."""
from pathlib import Path
from dataclasses import replace
import math
import runpy
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_reduction.direct_gaussian_joint import JointChart, fit_joint_from_gaussian_tracks, solve_weighted_twist
from joint_reduction.joint_information import information_inverse
from joint_reduction.joint_uncertainty import cluster_robust_covariance, curvature_covariance, curvature_evidence, heldout_predictive_gain


def track_case(curvature=.4, frames=6):
    rng=torch.Generator().manual_seed(2)
    x=torch.randn(30,3,generator=rng,dtype=torch.float64)*.2
    c=x.mean(0); L=float((x-c).square().sum(1).mean().sqrt())
    chart=JointChart(c,torch.tensor([1.,0.,0.]).double(),torch.tensor([0.,0.,curvature]).double(),torch.linspace(0,.4,frames).double(),L)
    tracks=torch.stack([x@chart.transform(t)[0].T+chart.transform(t)[1] for t in range(frames)])
    F=torch.diag(torch.tensor([100.,200.,50.]).double()).repeat(frames,len(x),1,1)
    return x,chart,tracks,F


def test_track_geometry():
    for curvature in [0.,.4,.01]:
        x,q,y,F=track_case(curvature)
        initial=q.retract(torch.tensor([.03,-.02,.015,.02]+[.015]*(len(y)-1)).double())
        fit=fit_joint_from_gaussian_tracks(x,y,F,initial=initial,max_iters=35)
        prediction=torch.stack([x@fit['chart'].transform(t)[0].T+fit['chart'].transform(t)[1] for t in range(len(y))])
        assert float((prediction-y).abs().max())<2e-7, (curvature,fit['status'],float((prediction-y).abs().max()))
        assert abs(fit['kappa']-curvature)<1e-6
        if curvature==0:
            fp=fit_joint_from_gaussian_tracks(x,y,F,initial=q,fixed_curvature=True)
            assert fp['energy']<1e-20
    x,q,y,F=track_case()
    shifted=replace(q,c=q.c+1000*q.length)
    a=fit_joint_from_gaussian_tracks(x,y,F,initial=q)
    b=fit_joint_from_gaussian_tracks(x+1000*q.length,y+1000*q.length,F,initial=shifted)
    assert abs(a['kappa']-b['kappa'])<1e-9
    assert abs(a['energy']-b['energy'])<1e-15
    finite=replace(q,finite_axis=True,eta=torch.zeros_like(q.eta),s=q.s*2)
    target=torch.stack([x@finite.transform(t)[0].T+finite.transform(t)[1] for t in range(len(q.s))])
    perturbed=finite.retract(torch.tensor([.01,-.02,.03,.04]+[.01]*(len(q.s)-1)).double())
    recovered=fit_joint_from_gaussian_tracks(x,target,F,initial=perturbed)
    assert recovered['finite_axis_fallback'] and recovered['energy']<1e-16


def test_minimal_kinematic_jacobian():
    for curvature in [0.,.4]:
        x,q,y,F=track_case(curvature)
        G=q.pose_jacobian(x,3)[:,:3]
        eps=1e-6
        for k in range(G.shape[-1]):
            v=torch.zeros(G.shape[-1]).double();v[k]=eps
            Rp,pp=q.transform(3,v);Rm,pm=q.transform(3,-v)
            numerical=((x@Rp.T+pp)-(x@Rm.T+pm))/(2*eps)
            assert torch.allclose(G[:,:,k],numerical,atol=2e-10,rtol=2e-8)
        assert q.dimension==4+len(y)-1
    x,q,y,F=track_case()
    finite=replace(q,finite_axis=True,eta=torch.zeros(3).double())
    assert torch.isfinite(finite.pose_jacobian(x,3)).all()


def test_weighted_twist():
    x,q,y,F=track_case()
    xi=torch.tensor([.2,.3,-.1,.1,-.2,.3]).double()
    v=xi[:3]+torch.linalg.cross(xi[3:].expand_as(x),x)
    fit=solve_weighted_twist(x,v,F[1])
    assert torch.allclose(fit['xi'],xi,atol=1e-12)


def test_cluster_uncertainty():
    A=torch.eye(7).double()*20
    rng=torch.Generator().manual_seed(42)
    scores=torch.randn(16,7,generator=rng).double()
    a=cluster_robust_covariance(A,scores)
    duplicated=cluster_robust_covariance(A*8,scores*8)
    assert torch.allclose(a['covariance'],duplicated['covariance'],atol=1e-15)
    bad=cluster_robust_covariance(A,scores*5)
    assert torch.allclose(bad['covariance'],25*a['covariance'])
    assert not cluster_robust_covariance(A,scores[:1])['estimable']
    low=information_inverse(torch.diag(torch.tensor([1.,0.,0.]).double()))
    assert low['rank']==1


def test_nested_evidence():
    curvature={'rank_beta':2,'estimable':True,'Q_eta':.5,'curvature_snr':math.sqrt(.5)}
    pred=heldout_predictive_gain([{'gain_total':v} for v in [-1.,1.,-.5,.5]])
    assert curvature_evidence(curvature,pred,authoritative=True)['status']=='UNRESOLVED_GEOMETRY'
    R=curvature_evidence({**curvature,'Q_eta':100.},heldout_predictive_gain([{'gain_total':v} for v in [5.,5.1,4.9,5.]]),authoritative=True)
    assert R['selected']=='revolute'
    P=curvature_evidence(curvature,heldout_predictive_gain([{'gain_total':v} for v in [-5.,-5.1,-4.9,-5.]]),authoritative=True)
    assert P['status']=='PRISMATIC_AT_RESOLUTION'
    assert curvature_evidence(curvature,pred,model_mismatch=True)['status']=='MODEL_MISMATCH'


def gpu_fixture():
    tests=runpy.run_path(str(Path(__file__).with_name('test_kpart_window.py')))
    st,labels=tests['two_region'](n_each=12)
    cams=[tests['camera'](width=40,height=40,tz=v) for v in [0.,.2,-.2]]
    x=st.mu[labels==1].double();c=x.mean(0)
    L=float((x-c).square().sum(1).mean().sqrt())
    q=JointChart(c,torch.tensor([1.,0.,0.],device='cuda').double(),torch.tensor([0.,0.,1.],device='cuda').double(),torch.linspace(0,.06,4,device='cuda').double(),L)
    from estimator.sparse_iekf import IEKFConfig
    from joint_reduction.kpart_window import WindowObjective
    obj=WindowObjective(st.mu,st.R,cams,[],IEKFConfig(robust_measurement=False,depth_sigma_model='constant:0.02:0.0'),state_template=st)
    from joint_reduction.direct_rgbd_joint import RGBDJointProblem
    problem=RGBDJointProblem(obj,labels==1,block_frames=2)
    frames=[]
    for t in range(len(q.s)):
        state=problem.state(q,t)
        frames.append((tests['obs_of'](state,state.mu,state.R,cams),t))
    obj.frames=frames
    return problem,q,st,labels


def test_renderer_chain_and_fit():
    from joint_reduction.direct_rgbd_joint import fit_joint_from_rgbd, measurement_terms
    problem,q,st,labels=gpu_fixture()
    step=torch.zeros(q.dimension,device='cuda').double();step[0]=.04;step[2]=.02;step[4:]=.02
    initial=q.retract(step)
    A,g,details=problem.linearize(initial)
    eps=5e-4
    for k in [0,2,4]:
        h=torch.zeros(q.dimension,device='cuda').double();h[k]=eps
        Ep=sum(r['loss'] for r in problem.evaluate(initial.retract(h)))
        Em=sum(r['loss'] for r in problem.evaluate(initial.retract(-h)))
        fd=(Ep-Em)/(2*eps)
        assert abs(fd+2*float(g[k]))<.08*max(abs(fd),1.),(k,fd,float(g[k]))
    fit=fit_joint_from_rgbd(problem,initial,max_iters=20,diagnostics=True)
    assert fit['energy']<sum(r['loss'] for r in problem.evaluate(initial))*.01,(fit['energy'],fit['status'])
    assert fit['orientation_renderer_jacobian']
    assert len(fit['details']['blocks'])==6
    assert all(torch.isfinite(row['F_pose']).all() for row in fit['details']['gaussian'])
    # Flags and nonfinite sensor holes are handled in the new shared measurement path.
    from estimator.renderer_bridge import build_linearization
    lin=build_linearization(st,[problem.objective.cameras[0]])
    obs=problem.objective.frames[0][0][0]
    depth=obs[1].clone();depth[0,0]=float('nan')
    cfg=replace(problem.objective.cfg,use_rgb=False,use_depth=False)
    p,r,e,w=measurement_terms(lin.views[0],(obs[0],depth,obs[2]),cfg)
    assert sum(e)==0 and all(torch.isfinite(v).all() for v in r)
    assert float(p[0].sum()+p[1].sum())==0


def test_source_contract():
    import ast
    root=Path(__file__).resolve().parents[1]/'joint_reduction'
    for name in ['direct_gaussian_joint.py','direct_rgbd_joint.py','joint_information.py','joint_uncertainty.py']:
        source=(root/name).read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.If):
                assert not any(v in ast.unparse(node.test).lower() for v in ['storage','usb','blade','stapler','fridge'])
        assert 'sqrt(N' not in source and 'reliable=True' not in source


def test_unobserved_and_duplicated_tracks():
    x,q,y,F=track_case()
    zero=fit_joint_from_gaussian_tracks(x,y,torch.zeros_like(F),max_iters=4)
    assert torch.isfinite(zero['chart'].d).all()
    assert information_inverse(zero['information'])['rank']==0
    # Repeating identical motion evidence scales both bread and block scores.
    a=fit_joint_from_gaussian_tracks(x,y,F,initial=q)
    b=fit_joint_from_gaussian_tracks(x.repeat(5,1),y.repeat(1,5,1),F.repeat(1,5,1,1),initial=q)
    sa=torch.stack([r['score'] for r in a['details']['blocks']])
    sb=torch.stack([r['score'] for r in b['details']['blocks']])
    assert torch.allclose(a['chart'].eta,b['chart'].eta)
    assert torch.allclose(cluster_robust_covariance(a['information'],sa)['covariance'],
                          cluster_robust_covariance(b['information'],sb)['covariance'],atol=1e-20)
    # Duplicated invisible tracks carry zero image information.
    extra=x[:3]+torch.tensor([100.,0.,0.])
    invisible=fit_joint_from_gaussian_tracks(torch.cat([x,extra]),
        torch.cat([y,extra[None].repeat(len(y),1,1)],1),
        torch.cat([F,torch.zeros(len(y),3,3,3).double()],1),initial=q)
    assert torch.allclose(a['chart'].eta,invisible['chart'].eta)
    assert torch.allclose(a['information'],invisible['information'])


def test_short_arc_covariance():
    x,q,y,F=track_case(curvature=.4,frames=12)
    q=replace(q,s=q.s*.0005)
    clean=torch.stack([x@q.transform(t)[0].T+q.transform(t)[1] for t in range(len(q.s))])
    rng=torch.Generator().manual_seed(19)
    observed=clean+torch.randn(clean.shape,generator=rng).double()*q.length*.002
    observed[0]=x
    R=fit_joint_from_gaussian_tracks(x,observed,F,initial=q,max_iters=40)
    P=fit_joint_from_gaussian_tracks(x,observed,F,initial=q,fixed_curvature=True,max_iters=40)
    alt=fit_joint_from_gaussian_tracks(x,observed,F,initial=replace(P['chart'],fixed_curvature=False),max_iters=40)
    if alt['energy']<R['energy']:R=alt
    assert R['energy'] <= P['energy']*(1+1e-10)
    scores=torch.stack([r['score'] for r in R['details']['blocks']])
    u=cluster_robust_covariance(R['information'],scores)
    curv=curvature_covariance(R['chart'],u['covariance'],R['information'])
    # Independent held-out observations, frozen train-fitted geometry and q.
    held=clean+torch.randn(clean.shape,generator=rng).double()*q.length*.002
    gains=[]
    for t in range(1,len(held)):
        def loss(chart):
            rot,p=chart.transform(t);r=held[t]-(x@rot.T+p)
            return float(torch.einsum('ni,nij,nj->',r,F[t],r))
        gains.append({'gain_total':loss(P['chart'])-loss(R['chart'])})
    predictive=heldout_predictive_gain(gains)
    decision=curvature_evidence(curv,predictive,authoritative=True)
    assert q.eta.norm()>0
    assert curv['Q_eta'] is None or curv['Q_eta']<decision['Q_critical'],curv
    assert predictive['ci'][0] <= 0 <= predictive['ci'][1],predictive
    assert decision['status']=='UNRESOLVED_GEOMETRY',decision


def test_renderer_information_reference():
    from estimator.renderer_bridge import build_linearization
    from joint_reduction.direct_rgbd_joint import measurement_terms
    from joint_reduction.joint_information import accumulate_gaussian_information, alpha_responsibility
    problem,q,st,labels=gpu_fixture()
    t=2;state=problem.state(q,t)
    lin=build_linearization(state,[problem.objective.cameras[0]])
    p,r,e,w=measurement_terms(lin.views[0],problem.objective.frames[t][0][0],problem.objective.cfg)
    F=accumulate_gaussian_information(lin,[p],[r])['cameras'][0]['F_pose']
    i=int((labels==1).nonzero()[0]);cols=[]
    for k in range(6):
        v=torch.zeros(len(st.mu),6,device='cuda');v[i,k]=1.
        rgb,depth=lin.apply_H(v)[0];cols.append(torch.cat([rgb.flatten(),depth.flatten()]).double())
    C=torch.stack(cols,1);precision=torch.cat([p[0].flatten(),p[1].flatten()]).double()
    reference=C.T@(precision[:,None]*C)
    assert torch.allclose(F[i],reference,atol=1e-4,rtol=2e-4),(F[i],reference)
    # Off-screen Gaussians have zero ownership and analytic Fisher.
    invisible=replace(st,mu=st.mu.clone());invisible.mu[0,0]=1e4
    hidden=build_linearization(invisible,[problem.objective.cameras[0]])
    fh=hidden.build_block_fisher([p])
    assert fh[0].abs().max()==0
    assert alpha_responsibility(invisible,problem.objective.cameras[0])[0]==0
    # Independent CUDA VJP accumulation agrees with the pixel-space normal.
    G=q.pose_jacobian(st.mu[labels==1],t)
    full=G.new_zeros(len(st.mu),6,G.shape[-1]);full[labels==1]=G
    from joint_reduction.direct_rgbd_joint import build_joint_renderer_jacobian
    HG=build_joint_renderer_jacobian(lin,full)
    C=torch.stack([torch.cat([v[0][0].flatten(),v[0][1].flatten()]) for v in HG],1).double()
    Href=C.T@(precision[:,None]*C)
    normal=[]
    for column in HG:
        ht=lin.apply_HT([(column[0][0]*p[0],column[0][1]*p[1])]).double()
        normal.append(torch.einsum('nik,ni->k',full,ht))
    Hcuda=torch.stack(normal,1)
    assert torch.allclose(Href,Hcuda,atol=1e-4,rtol=2e-4)


def test_textureless_rgbd_information():
    from joint_reduction.direct_rgbd_joint import RGBDJointProblem,fit_joint_from_rgbd
    from joint_reduction.kpart_window import WindowObjective
    problem,q,st,labels=gpu_fixture()
    st=replace(st,colors=torch.zeros_like(st.colors))
    fixture=runpy.run_path(str(Path(__file__).with_name('test_kpart_window.py')))
    cfg=replace(problem.objective.cfg,use_depth=False)
    obj=WindowObjective(st.mu,st.R,problem.objective.cameras,[],cfg,state_template=st)
    low=RGBDJointProblem(obj,labels==1)
    obj.frames=[(fixture['obs_of'](low.state(q,t),low.state(q,t).mu,low.state(q,t).R,obj.cameras),t) for t in range(len(q.s))]
    fit=fit_joint_from_rgbd(low,q,max_iters=2,diagnostics=False)
    assert information_inverse(fit['information'])['rank']==0
    decision=curvature_evidence(fit['curvature'],{'estimable':False},authoritative=True)
    assert decision['status']=='UNRESOLVED_GEOMETRY'


def test_rgbd_gauge_and_heldout_freeze():
    from estimator.manifold import so3_exp
    from joint_reduction.kpart_window import WindowObjective
    from joint_reduction.direct_rgbd_joint import RGBDJointProblem, compare_candidates
    problem,q,st,labels=gpu_fixture()
    delta=torch.zeros(q.dimension,device='cuda').double();delta[0]=.04;delta[4:]=.01
    fitted=q.retract(delta)
    held=RGBDJointProblem(problem.objective,labels==1,camera_ids=[2])
    before=[v.clone() for v in (fitted.d,fitted.eta,fitted.s,st.colors,st.scales,st.opacity)]
    rows=compare_candidates(held,replace(fitted,fixed_curvature=True,eta=fitted.eta*0),fitted)
    assert len(rows)==len(q.s) and all(r['camera']==2 for r in rows)
    assert all(torch.equal(a,b) for a,b in zip(before,(fitted.d,fitted.eta,fitted.s,st.colors,st.scales,st.opacity)))
    S=so3_exp(torch.tensor([[.3,-.2,.1]],device='cuda').double())[0]
    shift=torch.tensor([100.,-70.,45.],device='cuda').double()*q.length
    T=torch.eye(4,device='cuda').double();T[:3,:3]=S;T[:3,3]=shift
    inverse=torch.linalg.inv(T).T.float()
    cams=[replace(c,viewmatrix=(inverse@c.viewmatrix).contiguous(),
                  projmatrix=(inverse@c.projmatrix).contiguous()) for c in problem.objective.cameras]
    moved=replace(st,mu=(st.mu.double()@S.T+shift).float(),R=(S[None]@st.R.double()).float())
    obj=WindowObjective(moved.mu,moved.R,cams,problem.objective.frames,problem.objective.cfg,state_template=moved)
    other=RGBDJointProblem(obj,labels==1)
    chart=replace(fitted,c=S@fitted.c+shift,d=S@fitted.d,eta=S@fitted.eta)
    a=sum(r['loss'] for r in problem.evaluate(fitted));b=sum(r['loss'] for r in other.evaluate(chart))
    assert abs(a-b)<.004*max(a,b,1.),(a,b)
    assert abs(chart.eta.norm()-fitted.eta.norm())<1e-12


def test_correlated_camera_scores():
    from joint_reduction.direct_rgbd_joint import fit_joint_from_rgbd
    problem,q,st,labels=gpu_fixture()
    clean=fit_joint_from_rgbd(problem,q,max_iters=3,diagnostics=False)
    obj=problem.objective
    obj.frames=[([(rgb+(.02 if ci==0 else -.01),dep,valid) for ci,(rgb,dep,valid) in enumerate(obs)],fid)
                for obs,fid in obj.frames]
    biased=fit_joint_from_rgbd(problem,q,max_iters=12,diagnostics=False)
    assert float(biased['uncertainty']['covariance'].trace()) > float(clean['uncertainty']['covariance'].trace())
    scores=torch.stack([r['score'] for r in biased['details']['cameras']])
    assert float(scores.var(0).sum())>0
    assert all(torch.isfinite(c['delta']).all() for c in biased['camera_influence'])


def test_gt_loader_firewall():
    import tempfile
    from unittest.mock import patch
    from scene.dataset_readers import fetchPly, storePly
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        path=str(Path(tmp)/'points3D.ply')
        storePly(path,np.zeros((3,3)),np.ones((3,3))*100)
        with patch('scene.dataset_readers._load_matching_part_indices',side_effect=AssertionError('GT accessed')):
            pcd=fetchPly(path,load_gt_parts=False)
            assert pcd.part_indices is None


def test_camera_observation_fold_matching():
    from tools.run_iekf_scene import split_camera_fold
    cameras=list(range(8));frames=[([f'{t}/{c}' for c in cameras],t) for t in range(3)]
    train,obs,held=split_camera_fold(cameras,frames,[1,3,5,7])
    assert train==[1,3,5,7] and held['camera_ids']==[0,2,4,6]
    for rows,ids in [(obs,train),(held['frames'],held['camera_ids'])]:
        assert all(values==[f'{t}/{c}' for c in ids] for values,t in rows)


def test_correlated_rotational_tracks():
    from estimator.manifold import so3_exp
    x,q,y,F=track_case(frames=12)
    rng=torch.Generator().manual_seed(5)
    clean=y+torch.randn(y.shape,generator=rng).double()*1e-5
    biased=clean.clone()
    rot=so3_exp(torch.tensor([[.005,0.,0.]]).double())[0]
    for t in [2,3,7,8]:biased[t]=(biased[t]-q.c)@rot.T+q.c
    fits=[]
    for tracks,repeats in [(clean,1),(biased,1),(biased,6)]:
        fit=fit_joint_from_gaussian_tracks(x.repeat(repeats,1),tracks.repeat(1,repeats,1),
            F.repeat(1,repeats,1,1),initial=q,max_iters=35)
        scores=torch.stack([row['score'] for row in fit['details']['blocks']])
        fits.append(cluster_robust_covariance(fit['information'],scores)['covariance'])
    assert fits[1][2:4,2:4].trace()>fits[0][2:4,2:4].trace()*10
    assert torch.allclose(fits[1],fits[2],atol=1e-10,rtol=1e-6)


def test_rgbd_nested_families():
    from joint_reduction.direct_rgbd_joint import fit_joint_from_rgbd
    from joint_reduction.continuous_joint_geometry import fit_continuous_joint
    from joint_reduction.gaussian_transport_metric import canonical_moments
    fixture=runpy.run_path(str(Path(__file__).with_name('test_kpart_window.py')))
    for kappa in [0.,1.,.01]:
        problem,q,st,labels=gpu_fixture()
        q=replace(q,eta=q.eta*kappa)
        frames=[];tracks=[];rotations=[];translations=[]
        for t in range(len(q.s)):
            state=problem.state(q,t);tracks.append(state.mu[labels==1].double())
            R,p=q.transform(t);rotations.append(R);translations.append(p)
            frames.append((fixture['obs_of'](state,state.mu,state.R,problem.objective.cameras),t))
        problem.objective.frames=frames;x=st.mu[labels==1].double()
        moments=canonical_moments(x)
        A=fit_continuous_joint(q.c,torch.stack(rotations),torch.stack(translations),q.length,moments=moments)
        B=fit_joint_from_gaussian_tracks(x,torch.stack(tracks),torch.eye(3,device='cuda').double().repeat(len(frames),len(x),1,1),initial=q)
        P=fit_joint_from_rgbd(problem,q,fixed_curvature=True,max_iters=20,diagnostics=False)
        R=fit_joint_from_rgbd(problem,q,max_iters=20,diagnostics=False)
        assert R['energy']<=P['energy']+1e-5,(kappa,R['energy'],P['energy'])
        assert abs(A['kappa']-kappa)<1e-4 and abs(B['kappa']-kappa)<1e-5
        assert abs(R['kappa']-kappa)<1e-4
        if kappa>0:assert P['energy']>R['energy']


def test_information_observer_is_read_only():
    from argparse import Namespace
    from estimator.supernodes import build_supernodes
    from tools.kpart_pipeline import bootstrap_positions
    from tools.direct_joint_pipeline import motion_image_information
    problem,q,st,labels=gpu_fixture()
    cfg=replace(problem.objective.cfg,max_iters=2,pcg_max_iters=8)
    args=Namespace(supernode_size=6,seed=0,dt=1/30,bootstrap_iters=2)
    partition=build_supernodes(st.mu,target_size=6,seed=0)
    baseline,_=bootstrap_positions(st,problem.objective.cameras,problem.objective.frames,cfg,partition,args,verbose=False)
    captured=[]
    def observe(t,state,prior=None):
        captured.append((motion_image_information(state,problem.objective.cameras,problem.objective.frames[t][0],cfg),prior))
    diagnostic,_=bootstrap_positions(st,problem.objective.cameras,problem.objective.frames,cfg,partition,args,verbose=False,observer=observe)
    assert all(torch.equal(a,b) for a,b in zip(baseline,diagnostic))
    assert captured[0][1] is None and len(captured)==len(q.s)
    assert all(torch.isfinite(p).all() and p.shape==(len(st.mu),3,3) for _,p in captured[1:])


if __name__=='__main__':
    tests=[test_track_geometry,test_minimal_kinematic_jacobian,test_weighted_twist,test_cluster_uncertainty,test_nested_evidence,test_source_contract,test_unobserved_and_duplicated_tracks,test_short_arc_covariance,test_gt_loader_firewall,test_camera_observation_fold_matching,test_correlated_rotational_tracks]
    if torch.cuda.is_available():tests += [test_renderer_chain_and_fit,test_renderer_information_reference,
        test_rgbd_gauge_and_heldout_freeze,test_correlated_camera_scores,test_rgbd_nested_families,
        test_textureless_rgbd_information,test_information_observer_is_read_only]
    for test in tests:
        print('START',test.__name__,flush=True);test();print('PASS',test.__name__,flush=True)
    print(len(tests),'direct joint tests passed')
