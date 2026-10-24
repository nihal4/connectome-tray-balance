"""Is ripeness visible in what the brain gets? Drive the hand to 12 and 8 cm in front of the truss on N layouts
and fit a linear read-out (logistic regression, 5-fold) of "is this tomato ripe" from each sense.
Usage: python scripts/info_pickripe.py FACETS N [rgbd]   (rgbd: the wrist depth camera instead of the wrist eye)"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")      # off-screen rendering (no display needed)
import sys, numpy as np, time
sys.path.insert(0,'.')
from ctb.envs import build_env
import ctb.envs.panda as P
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
nf=int(sys.argv[1]); N=int(sys.argv[2])
env=build_env('panda_pickripe',0,grip_mode='absolute',eye_adapt=True,n_facets=nf,**({'wrist_cam':sys.argv[3]} if len(sys.argv)>3 else {}))
res={0.12:[[],[]],0.08:[[],[]]}
t0=time.time()
for i in range(N):
    o,_=env.reset(seed=8000+i)
    mid=env.data.mocap_pos[env.plant_mocap]+np.array([-P.BRANCH_LEN,0,-P.PED_LEN-P.TOM_R])
    # drive the hand straight to the look points (no teacher randomness), record obs at each
    for back in (0.12,0.08):
        goal=mid-back*P.APPROACH
        for t in range(120):
            a=np.zeros(4); a[3]=P.PICK_OPEN; a[:3]=np.clip((goal-env.tcp_pos())/P.STEP*0.6,-1,1)
            o,*_=env.step(a)
            if np.linalg.norm(env.tcp_pos()-goal)<0.01 and np.linalg.norm(env.tcp_vel())<0.05: break
        lab=[k for k in range(3) if env.ripe[k]==2]
        y=min(lab,key=lambda k:abs(P.TRUSS_Y[k]))
        res[back][0].append(o[:env.n_brain_obs]); res[back][1].append(tuple(r==2 for r in env.ripe))
nv=env.n_vis; ns=env.n_smell; F=nf
for back,(X,Y) in res.items():
    X=np.array(X); Yr=np.array(Y)
    # per tomato: is it ripe? (3 binary tasks), averaged
    for name,sl in (('wrist eye',slice(F*3,nv)),('head eye',slice(0,F*3)),('smell',slice(X.shape[1]-ns,X.shape[1])),('all',slice(0,X.shape[1]))):
        accs=[cross_val_score(LogisticRegression(C=0.5,max_iter=4000),X[:,sl],Yr[:,k],cv=5).mean() for k in range(3)]
        print(f'facets {nf} | {int(back*100)} cm in front | {name:9s}: is each tomato ripe? {np.mean(accs):.0%} (slots {[round(a,2) for a in accs]})')
print('chance', round(np.mean([max(Yr[:,k].mean(),1-Yr[:,k].mean()) for k in range(3)]),2), time.time()-t0)
