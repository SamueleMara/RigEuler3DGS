import math
import torch
from RigEuler3DGS_first_return_reference import first_return_ray

torch.set_default_dtype(torch.float64)

def run(D, z, alpha, lam, s=0.002):
    z=torch.tensor(z,dtype=torch.float64)
    a=torch.tensor(alpha,dtype=torch.float64)
    return first_return_ray(torch.tensor(D),z,a,torch.full_like(z,s),z_near=0.3,z_far=1.3,lambda_unknown=lam,sigma_unknown=s)

def test_first_hit_normalization():
    for z,a,lam in [([],[],0.4), ([.7],[.6],0.0), ([.7],[.6],2.0),([.7,.9],[.3,.9],0.8),([.9,.7],[.9,.3],4.0)]:
        r=run(.85,z,a,lam)
        mass=r.gaussian_first_hit_mass.sum()+r.unknown_first_hit_mass+r.no_return_mass
        assert torch.allclose(mass,torch.tensor(1.0),rtol=1e-11,atol=1e-11),mass
        assert torch.isfinite(r.log_valid_depth_likelihood)
        assert torch.allclose(r.gamma_gaussian.sum()+r.gamma_unknown,torch.tensor(1.0),atol=1e-10)

def test_hidden_cavity_not_associated_to_door_depth():
    r=run(.75,[.75,.95],[.9,.9],.25,s=.004)
    assert r.gamma_gaussian[0]>0.99
    assert r.gamma_gaussian[1]<1e-7

def test_false_front_blocks_unknown_surface():
    reference=run(.95,[],[],.8,s=.003)
    wrong=run(.95,[.75],[.9],.8,s=.003)
    # a wrong opaque foreground geometry makes the same deeper unknown measurement
    # about ten times less likely, while the predicted hit at .75 fails the depth match
    ratio=(wrong.log_valid_depth_likelihood-reference.log_valid_depth_likelihood).exp()
    assert abs(float(ratio)-.1)<.002,ratio

def test_gaussian_hit_probability_without_unknown():
    r=run(.7,[.7],[.8],0,s=.005)
    assert torch.allclose(r.gaussian_first_hit_mass,torch.tensor([.8]))
    assert torch.allclose(r.no_return_mass,torch.tensor(.2))
    assert torch.allclose(r.gamma_gaussian,torch.tensor([1.0]))

def test_depth_translation_gradient_finite_difference():
    z=torch.tensor([.74,.95],requires_grad=True)
    a=torch.tensor([.55,.73],requires_grad=True)
    D=torch.tensor(.78)
    def energy(z,a):
        return -first_return_ray(D,z,a,torch.tensor([.008,.008]),z_near=.3,z_far=1.3,lambda_unknown=1.2,sigma_unknown=.007).log_valid_depth_likelihood
    E=energy(z,a)
    gz,ga=torch.autograd.grad(E,(z,a))
    h=1e-6
    for i in range(2):
        dz=torch.zeros_like(z);dz[i]=h
        gfd=(energy(z.detach()+dz,a.detach())-energy(z.detach()-dz,a.detach()))/(2*h)
        assert torch.allclose(gfd,gz[i],atol=3e-4,rtol=1e-4),(gfd,gz[i])
        da=torch.zeros_like(a);da[i]=h
        gfd=(energy(z.detach(),a.detach()+da)-energy(z.detach(),a.detach()-da))/(2*h)
        assert torch.allclose(gfd,ga[i],atol=3e-4,rtol=1e-4),(gfd,ga[i])

def test_no_rendered_geometry_unknown_branch():
    r=run(.95,[],[],1.0,s=.003)
    assert r.gamma_unknown==1
    assert r.unknown_first_hit_mass>0
    assert r.no_return_mass>0


def test_coincident_depths_have_zero_unknown_interval():
    r=run(.75,[.75,.75,.95],[.2,.3,.7],.4)
    assert torch.isfinite(r.log_valid_depth_likelihood)
    assert torch.allclose(r.gaussian_first_hit_mass.sum()+r.unknown_first_hit_mass+r.no_return_mass,
                          torch.tensor(1.0),atol=1e-11)


def test_sorting_invariance_for_distinct_contributors():
    a=run(.751,[.95,.75],[.5,.3],1.7)
    b=run(.751,[.75,.95],[.3,.5],1.7)
    assert torch.allclose(a.log_valid_depth_likelihood,b.log_valid_depth_likelihood)
    assert torch.allclose(a.gamma_gaussian,b.gamma_gaussian.flip(0))


def test_many_random_rays_mass_normalizes():
    g=torch.Generator().manual_seed(1)
    for k in range(0,9):
        z=.31+.98*torch.rand(k,generator=g)
        a=.01+.98*torch.rand(k,generator=g)
        rr=first_return_ray(torch.tensor(.8),z,a,torch.full_like(z,.01),
                             z_near=.3,z_far=1.3,lambda_unknown=1.2,sigma_unknown=.02)
        total=rr.gaussian_first_hit_mass.sum()+rr.unknown_first_hit_mass+rr.no_return_mass
        assert abs(float(total)-1)<1e-10
