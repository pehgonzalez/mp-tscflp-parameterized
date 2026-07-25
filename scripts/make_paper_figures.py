#!/usr/bin/env python3
# Result figures for the MP-TSCFLP paper. Reads results/q1_xp.csv,
# results/mauri_kstar.csv and results/mauri_mip.csv (real campaign data) and
# writes PDF+PNG to paper/figures/. Styling follows the paper's TikZ figures:
# Computer Modern serif mathtext, no top/right spines, axis arrows, direct
# labels at line ends where feasible, legends outside the axes area, colour
# only where identity demands it (the overlapping k* and size-group scatters,
# Okabe-Ito hues + distinct markers so the encoding survives greyscale and CVD).
# The plotted numbers are identical to the previous revision of this script.
import csv, statistics as st, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
OUT=os.path.join(RES,"..","paper","figures")
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({
    "font.family":"serif",
    "font.serif":["cmr10","Computer Modern Roman","DejaVu Serif"],
    "mathtext.fontset":"cm",
    "axes.unicode_minus":False,
    "axes.formatter.use_mathtext":True,
    "font.size":9,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
    "axes.linewidth":0.6,
    "xtick.direction":"in","ytick.direction":"in","legend.frameon":False,
    "axes.spines.top":False,"axes.spines.right":False})
KS=[4,8,12,16]
COL={4:"#0072B2",8:"#E69F00",12:"#009E73",16:"#CC79A7"}
MK ={4:"o",8:"s",12:"^",16:"D"}

def axis_arrows(ax):
    """Arrow tips on the two remaining spines, exemplar style."""
    ax.plot(1.0,0,">",transform=ax.transAxes,clip_on=False,color="black",
            ms=3.5,zorder=6)
    ax.plot(0,1.0,"^",transform=ax.transAxes,clip_on=False,color="black",
            ms=3.5,zorder=6)

rows=list(csv.DictReader(open(os.path.join(RES,"q1_xp.csv"))))
def n_of(r): return int(r["nI"])+int(r["nJ"])
def kt(r): return int(r["kstar_target"])
BOUND=[r for r in rows if r["nK"]=="30" and r["nL"]=="3"]  # boundary/pilot family (has solves)

# ---------- Figure Q1: full regime picture + inverse k* effect ----------
fig,(axa,axb,axc)=plt.subplots(1,3,figsize=(7.1,2.9),
                           gridspec_kw={"width_ratios":[1.35,0.95,0.95]})
TL=60.0
# panel a: ALL 460 runs. Solved (colour+shape by k*) live only at n<=24;
# censored shown as grey open circles across the whole n range, so the wall of
# the 400-instance main grid (n>=40) is visible.
import random; rng=random.Random(3)
# censored first (background), jittered
xc=[];yc=[]
for r in rows:
    if r["status"]=="OPTIMAL": continue
    xc.append(n_of(r)+rng.uniform(-1.3,1.3)); yc.append(TL*rng.uniform(0.97,1.03))
axa.scatter(xc,yc,s=10,facecolors="none",edgecolors="0.6",linewidths=0.4,
            alpha=0.45,zorder=2)
# solved on top
for k in KS:
    xs=[];ys=[]
    for r in BOUND:
        if kt(r)==k and r["status"]=="OPTIMAL":
            xs.append(n_of(r)+(k-10)*0.16); ys.append(float(r["time"]))
    axa.scatter(xs,ys,c=COL[k],marker=MK[k],s=30,edgecolors="white",
                linewidths=0.4,zorder=4,label=f"$k^\\ast={k}$")
axa.axhline(TL,ls=(0,(4,3)),lw=0.7,color="0.35")
axa.text(82,TL*1.08,"60 s limit",va="bottom",ha="right",fontsize=7,color="0.35")
axa.axvspan(18,25,color="0.92",zorder=0)
axa.annotate("main grid\nall 400 censored",xy=(60,TL*0.9),
             xytext=(63,7),fontsize=7,ha="center",color="0.3",
             arrowprops=dict(arrowstyle="->",color="0.5",lw=0.6))
axa.set_yscale("log"); axa.set_xticks([20,30,40,60,80])
axa.set_xlim(15,84); axa.set_ylim(0.2,110)
axa.set_xlabel("$n=|I|+|J|$"); axa.set_ylabel("enumeration time (s)")
axa.set_title("(a)",fontsize=9,loc="left")
# legend outside the axes area, one row above the panel
axa.legend(loc="lower left",bbox_to_anchor=(0.07,1.0),ncol=4,fontsize=7,
           handletextpad=0.15,labelspacing=0.25,columnspacing=0.7,
           borderaxespad=0.0)
# panel b: median solve time vs k* for n=20 and n=24 (solved runs only),
# directly labelled at the line ends instead of a legend box. Points whose
# median covers fewer than three solved runs are drawn as open (unfilled)
# markers, determined programmatically from the data (F8).
endpts={}
for n,ls,mk in [(20,"-","o"),(24,(0,(4,2)),"s")]:
    xs=[];ys=[];full=[]
    for k in KS:
        ts=[float(r["time"]) for r in BOUND if n_of(r)==n and kt(r)==k and r["status"]=="OPTIMAL"]
        if ts:
            xs.append(k); ys.append(st.median(ts)); full.append(len(ts)>=3)
    axb.plot(xs,ys,ls=ls,lw=1.4,color="0.15",zorder=2)
    for x,y,fl in zip(xs,ys,full):
        axb.plot([x],[y],ls="none",marker=mk,ms=5,color="0.15",
                 mfc=("0.15" if fl else "white"),mew=1.0,zorder=3)
    endpts[n]=(xs[-1],ys[-1])
for n,(x,y) in endpts.items():
    axb.annotate(f"$n={n}$",xy=(x,y),xytext=(5,4),textcoords="offset points",
                 fontsize=8,ha="left",va="bottom",color="0.15")
axb.set_xticks(KS); axb.set_xlim(2.5,18.5)
axb.set_xlabel("covering bound $k^\\ast$")
axb.set_ylabel("median solve time (s)")
axb.set_title("(b)",fontsize=9,loc="left")
# panel c: solved fraction per size under the two budgets (60 s boundary
# family, 3 seeds/cell; 600 s budget family, 10 seeds/cell)
b600=list(csv.DictReader(open(os.path.join(RES,"q1_boundary600.csv"))))
NSC=[20,24,28,30,40]
frac60=[]; frac600=[]
for n in NSC:
    c60=[r for r in BOUND if n_of(r)==n]
    frac60.append(sum(1 for r in c60 if r["status"]=="OPTIMAL")/len(c60))
    c6=[r for r in b600 if int(r["nI"])+int(r["nJ"])==n]
    frac600.append(sum(1 for r in c6 if r["status"]=="OPTIMAL")/len(c6))
axc.plot(NSC,frac60,ls="-",lw=1.4,color="0.15",marker="o",ms=4.5,
         mfc="0.15",zorder=3)
axc.plot(NSC,frac600,ls=(0,(4,2)),lw=1.4,color="0.15",marker="s",ms=4.5,
         mfc="white",mew=1.0,zorder=3)
axc.text(21.4,0.72,"$60$ s",fontsize=8,color="0.15",ha="left")
axc.annotate("$600$ s",xy=(28,frac600[2]),xytext=(8,5),
             textcoords="offset points",fontsize=8,color="0.15")
axc.set_xticks([20,24,28,40]); axc.set_xlim(17,43); axc.set_ylim(-0.05,1.1)
axc.set_yticks([0,0.5,1.0])
axc.set_xlabel("$n=|I|+|J|$")
axc.set_ylabel("solved fraction")
axc.set_title("(c)",fontsize=9,loc="left")
axis_arrows(axa); axis_arrows(axb); axis_arrows(axc)
fig.tight_layout(pad=0.4,w_pad=1.2)
for ext in ("pdf","png"):
    fig.savefig(os.path.join(OUT,f"fig_q1_regime.{ext}"),dpi=200,bbox_inches="tight")
plt.close(fig)

# ---------- Figure Q3: k* distribution across benchmark groups (strip+median) ----------
m=list(csv.DictReader(open(os.path.join(RES,"mauri_kstar.csv"))))
def grp(r): return f"{'50' if r['nI']=='50' else '100'}-{r['nL']}"
order=["50-5","50-10","100-5","100-10"]
data={g:[] for g in order}
for r in m: data[grp(r)].append(int(r["kstar_py"]))
fig2,ax=plt.subplots(figsize=(4.6,3.1))
import random as _r; rng2=_r.Random(7)
# cores por grupo identicas as da Fig. do scatter (cor segue a entidade)
GCOL={"50-5":"#CC79A7","50-10":"#009E73","100-5":"#E69F00","100-10":"#0072B2"}
for i,g in enumerate(order):
    xs=[i+rng2.uniform(-0.16,0.16) for _ in data[g]]
    ax.scatter(xs,data[g],s=16,color=GCOL[g],alpha=0.85,edgecolors="white",
               linewidths=0.4,zorder=3)
    med=st.median(data[g]); lo=min(data[g]); hi=max(data[g])
    ax.plot([i-0.28,i+0.28],[med,med],color="0.1",lw=1.6,zorder=4)   # median
    ax.plot([i,i],[lo,hi],color="0.55",lw=0.7,zorder=1)              # range spine
    ax.annotate(f"{lo}-{hi}",xy=(i,hi),xytext=(0,3),textcoords="offset points",
                ha="center",fontsize=6.6,color="0.35")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
ax.set_xlim(-0.5,3.6); ax.set_ylim(0,76)
ax.set_xlabel("benchmark group $|I|$-$|L|$")
ax.set_ylabel("covering bound $k^\\ast$")
axis_arrows(ax)
fig2.tight_layout(pad=0.4)
for ext in ("pdf","png"):
    fig2.savefig(os.path.join(OUT,f"fig_q3_kstar.{ext}"),dpi=200,bbox_inches="tight")
plt.close(fig2)

# ---------- Figure Q3 scatter: terminal gap against k*, by size group ----------
# Same join and numbers as scripts/finalize_after_campaign.py::make_q3_scatter.
kstar={r["instance"]:r for r in m}
srows=[]
for r in csv.DictReader(open(os.path.join(RES,"mauri_mip.csv"))):
    name=r["instance"].strip()
    kr=kstar.get(name) or kstar.get(name+".txt") or kstar.get(name.replace(".txt",""))
    if not kr: continue
    try: g=float(r["gap"])
    except ValueError: continue
    srows.append(dict(k=float(kr["kstar_py"]),g=g,
                      grp=f"{kr['nI']}-{kr['nJ']}-{kr['nK']}-L{kr['nL']}"))
groups=sorted({r["grp"] for r in srows})
palette=["#0072B2","#E69F00","#009E73","#CC79A7","#D55E00"]
markers=["o","s","^","D","v"]
fig3,ax3=plt.subplots(figsize=(6.4,3.8))
for gi,g in enumerate(groups):
    pts=[r for r in srows if r["grp"]==g]
    ax3.scatter([r["k"] for r in pts],[100.0*r["g"] for r in pts],
                s=34,alpha=0.9,label=g.replace("-L", ", $|L|$="),
                color=palette[gi%len(palette)],
                marker=markers[gi%len(markers)],
                edgecolors="white",linewidths=0.8,zorder=3)
ax3.set_xlabel(r"covering bound $k^\ast$")
ax3.set_ylabel("terminal gap (%)")
ax3.grid(axis="y",alpha=0.25,linewidth=0.5,zorder=0)
# the four groups overlap, so the colour+marker legend stays, placed fully
# outside the axes area
ax3.legend(loc="lower left",bbox_to_anchor=(0.02,1.01),ncol=2,fontsize=7.5,
           handletextpad=0.2,labelspacing=0.3,columnspacing=1.2,
           borderaxespad=0.0,title="size group $|I|$-$|J|$-$|K|$",
           title_fontsize=7.5)
ax3._legend_ = ax3.get_legend()
ax3._legend_. _legend_box.align = "left"
axis_arrows(ax3)
fig3.tight_layout(pad=0.4)
for ext in ("pdf","png"):
    fig3.savefig(os.path.join(OUT,f"fig_q3_scatter.{ext}"),dpi=200,bbox_inches="tight")
plt.close(fig3)
print("figuras geradas:",sorted(os.listdir(OUT)))
