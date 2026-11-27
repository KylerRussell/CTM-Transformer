"""Figures for research/PAPER_DRAFT.md, drawn from committed results (2026-11-20).

Palette: the dataviz reference palette, light mode, categorical slots 1-3 (blue, orange, aqua; validated all-pairs
in light mode per the palette reference), neutral tokens for text, axes and gridlines. Aqua is below 3:1 contrast on
the surface, so every series is also direct-labeled.

Usage: python -m scripts.paper_figures   (writes research/figures/fig{1..5}_*.{png,pdf})
"""
import glob,json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('research/figures');R=Path('research/results')
SURFACE,TEXT,TEXT2,MUTED,GRID,AXIS='#fcfcfb','#0b0b0b','#52514e','#898781','#e1e0d9','#c3c2b7'
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
plt.rcParams.update({'figure.facecolor':SURFACE,'axes.facecolor':SURFACE,'savefig.facecolor':SURFACE,'font.size':9,'axes.edgecolor':AXIS,
                     'axes.labelcolor':TEXT2,'xtick.color':MUTED,'ytick.color':MUTED,'xtick.labelcolor':TEXT2,'ytick.labelcolor':TEXT2,'text.color':TEXT,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'grid.linestyle':'-',
                     'axes.axisbelow':True,'lines.linewidth':1.6,'lines.solid_capstyle':'round','lines.solid_joinstyle':'round','legend.frameon':False})
mean=lambda v:sum(v)/len(v)


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=220)
    plt.close(fig)


def fig1_fixed_point():
    """(a) Held-out loss against ticks, 1,024 windows (scripts/lm_unified_eval.py): five-seed means, band = range over
    the five seeds; T8 as a line (five-seed mean). (b) The state's relative change per tick."""
    U=R/'lm_unified';load=lambda n:json.loads((U/f'{n}.json').read_text())
    five=lambda a,first:first+[f'{a}_lr0.002_s{s}' for s in (3,4,5)]  # seeds 1234 and 2, then the revision's seeds 3-5
    series=[('Dans (RDT)',BLUE,five('Dans',['Dans_rand_lr0.002','Dans_rand_lr0.002_s2'])),('CTsq',ORANGE,five('CTsq',['CTsq_lr0.002','CTsq_lr0.002_s2'])),
            ('CTsqS',AQUA,five('CTsqS',['CTsqS_lr0.002','CTsqS_lr0.002_s2']))]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.6,3.4));t8=mean([load(n)['loss']['1'] for n in five('T8',['T8_lr0.002','T8_lr0.002_s2'])])
    ax.axhline(t8,color=MUTED,linewidth=1.2);ax.text(96,t8+0.006,'8-layer Transformer (one pass)',ha='right',va='bottom',color=TEXT2,fontsize=8)
    for name,color,runs in series:
        rs=[load(n) for n in runs];ticks=sorted(int(k) for k in rs[0]['loss']);ys=[[r['loss'][str(t)] for r in rs] for t in ticks]
        ax.fill_between(ticks,[min(v) for v in ys],[max(v) for v in ys],color=color,alpha=0.18,linewidth=0)
        ax.plot(ticks,[mean(v) for v in ys],color=color,label=name)
        ch=[mean([r['relative_state_change'][str(t)] for r in rs]) for t in range(2,97)];bx.plot(range(2,97),ch,color=color,label=name)  # tick 1 starts from the zero state
    for a in (ax,bx):a.set_xscale('log',base=2);a.set_xticks([1,2,4,8,16,32,64]);a.set_xticklabels(['1','2','4','8','16','32','64']);a.set_xlabel('Ticks')
    ax.set_xlim(0.9,110);ax.set_ylim(3.55,4.0);ax.set_ylabel('Held-out loss (nats per token)');ax.legend(loc='upper right',bbox_to_anchor=(1.0,0.9),fontsize=8,labelcolor=TEXT2)
    ax.set_title('(a) The loss stops improving by about 8 ticks',fontsize=9.5,color=TEXT,loc='left')
    bx.set_yscale('log');bx.set_ylabel('Relative change of the state per tick');bx.set_xlim(1.8,110);bx.set_xticks([2,4,8,16,32,64]);bx.set_xticklabels(['2','4','8','16','32','64'])
    bx.set_title('(b) The state keeps moving (about 0.2% per tick)',fontsize=9.5,color=TEXT,loc='left')
    save(fig,'fig1_fixed_point')


def fig2_serial_parity():
    """CTsqS seed 1 on parity: the first tick reaching 90% accuracy, by input length, with linear (intercept) and
    quadratic least-squares fits."""
    import numpy as np
    a=json.loads((R/'parity_ticks/analysis_CTsqS_seed1.json').read_text())['ticks_vs_length']
    pts=sorted((int(n),v['first_tick_90']) for n,v in a.items() if v['first_tick_90'] is not None);x=np.array([p[0] for p in pts],float);y=np.array([p[1] for p in pts],float)
    lin=np.polyfit(x,y,1);quad=np.polyfit(x,y,2);g=np.linspace(0,120,200)
    fig,ax=plt.subplots(figsize=(5.2,3.3));ax.axvspan(0,40,color=GRID,alpha=0.6,linewidth=0);ax.text(20,66,'trained lengths\n(1-40 bits)',ha='center',va='top',color=TEXT2,fontsize=8)
    ax.plot(g,np.polyval(lin,g),color=MUTED,linewidth=1.1,linestyle='--',label=f'linear fit ({lin[0]:.2f} ticks per bit)')
    ax.plot(g,np.polyval(quad,g),color=BLUE,linewidth=1.4,alpha=0.6,label='quadratic fit')
    ax.plot(x,y,'o',color=BLUE,markersize=6,markeredgecolor=SURFACE,markeredgewidth=1.5,label='measured')
    ax.legend(loc='lower right',fontsize=8,labelcolor=TEXT2)
    ax.set_xlim(0,122);ax.set_ylim(0,72);ax.set_xlabel('Input length (bits)');ax.set_ylabel('First tick reaching 90% accuracy')
    ax.set_title('A serial solution: thinking time grows with the input',fontsize=9.5,color=TEXT,loc='left');save(fig,'fig2_serial_parity')
    return lin,quad


def fig3_binding():
    """Share of runs that bind (ID query accuracy >= 90%), confirmatory seeds only: width 128, seeds 5-24 (CTsqS, Dans,
    T8 from the confirmation and replication phases; controls from phase 7); width 256, test seeds 5-14."""
    M=R/'recall_memory';ld=lambda f:json.loads((M/f).read_text())
    w128={}
    for c in ('T8','Dans','CTsqS'):w128[c]=[ld(f'confirm_{c}_cur_seed{s}.json') for s in range(5,15)]+[ld(f'replicate_{c}_cur_seed{s}.json') for s in range(15,25)]
    for c in ('DansS','DansT4','CTsq1S'):w128[c]=[ld(f'control_{c}_cur_seed{s}.json') for s in range(5,25)]
    best=json.loads((M/'BEST_LR_W256.json').read_text());w256={c:[ld(f'w256test_{c}_cur_lr{best[c]:g}_seed{s}.json') for s in range(5,15)] for c in ('T8','Dans','CTsqS')}
    k=lambda rs:sum(r['id']>=90 for r in rs)
    names={'T8':'8-layer\nTransformer','Dans':'Dans','DansT4':'Dans,\nsmall ticks','DansS':'Dans +\nsink','CTsq1S':'first-order\nquery + sink','CTsqS':'CTsqS'}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.6,3.5),gridspec_kw={'width_ratios':[2,1]})
    for a,data,order,color,title in ((ax,w128,('T8','Dans','DansT4','DansS','CTsq1S','CTsqS'),BLUE,'(a) Width 128, seeds 5-24'),(bx,w256,('T8','Dans','CTsqS'),ORANGE,'(b) Width 256, seeds 5-14')):
        ys=[100*k(data[c])/len(data[c]) for c in order];a.bar(range(len(order)),ys,width=0.62,color=color,edgecolor=SURFACE,linewidth=1.5)
        for i,c in enumerate(order):a.text(i,ys[i]+2,f'{k(data[c])}/{len(data[c])}',ha='center',va='bottom',color=TEXT2,fontsize=8)
        a.set_xticks(range(len(order)));a.set_xticklabels([names[c] for c in order],fontsize=7.5);a.set_ylim(0,108);a.set_yticks([0,25,50,75,100]);a.grid(axis='x',visible=False)
        a.set_title(title,fontsize=9.5,color=TEXT,loc='left')
    ax.set_ylabel('Runs that learn the binding (%)');save(fig,'fig3_binding')
    return {c:k(v) for c,v in w128.items()}


def fig5_time_to_bind():
    """Kaplan-Meier: share of runs bound by each update (ID check >= 90%, every 1,000 updates), seeds 5-24, width 128."""
    M=R/'recall_memory';ld=lambda f:json.loads((M/f).read_text());runs={}
    for c in ('T8','Dans','CTsqS'):runs[c]=[ld(f'confirm_{c}_cur_seed{s}.json') for s in range(5,15)]+[ld(f'replicate_{c}_cur_seed{s}.json') for s in range(15,25)]
    for c in ('DansS','DansT4','CTsq1S'):runs[c]=[ld(f'control_{c}_cur_seed{s}.json') for s in range(5,25)]
    style={'CTsq1S':(MUTED,'--','first-order query + sink'),'DansT4':(MUTED,':','Dans, small ticks'),'DansS':(TEXT2,'-.','Dans + sink'),
           'T8':(AQUA,'-','8-layer Transformer'),'CTsqS':(BLUE,'-','CTsqS'),'Dans':(ORANGE,'-','Dans')}  # coloured lines drawn last, on top
    fig,ax=plt.subplots(figsize=(5.6,3.4));ends=[]
    for c,(color,ls,label) in style.items():
        t=sorted(r['updates_to_90'] for r in runs[c] if r['updates_to_90']);xs=[0];ys=[0]
        for u in range(1000,10001,1000):xs.append(u);ys.append(100*sum(x<=u for x in t)/len(runs[c]))
        ax.step(xs,ys,where='post',color=color,linestyle=ls,linewidth=1.6);ends.append((ys[-1],label))
    ends.sort();last=-99
    for y,label in ends:
        y=max(y,last+6.5);last=y;ax.text(10150,y,label,va='center',color=TEXT2,fontsize=7.5)
    ax.set_xlim(0,13600);ax.set_xticks(range(0,10001,2000));ax.set_ylim(-2,102);ax.set_xlabel('Update');ax.set_ylabel('Runs bound by this update (%)')
    ax.set_title('Time to bind (width 128, seeds 5-24)',fontsize=9.5,color=TEXT,loc='left');save(fig,'fig5_time_to_bind')


def fig4_emergence():
    """In-context recall during continued pretraining at d704: ID accuracy at 16 ticks by update."""
    S=R/'recall_speed';fig,ax=plt.subplots(figsize=(5.2,3.3))
    for arm,color,name in (('Dans_sampler',BLUE,'Dans (RDT)'),('CTsqS_sampler',ORANGE,'CTsqS')):
        for k,style in ((1,'-'),(2,'--')):
            r=json.loads((S/f'{arm}_r{k}.json').read_text());steps=sorted(int(s) for s in r['snapshots']);y=[r['snapshots'][str(s)]['id'] for s in steps]
            ax.plot([0]+steps,[None]+y,style,color=color,label=f'{name}, replicate {k}')
    ax.axhline(80,color=MUTED,linewidth=1.0);ax.text(200,81.5,'emergence threshold (80%)',ha='right',va='bottom',color=TEXT2,fontsize=8)
    ax.set_xlim(0,200);ax.set_ylim(0,104);ax.set_xticks([0,25,50,75,100,125,150,175,200]);ax.set_xlabel('Update of continued pretraining (262,144 tokens each)');ax.set_ylabel('Recall accuracy, N 2-16 (%)')
    ax.legend(loc='lower right',fontsize=8,labelcolor=TEXT2);ax.set_title('Pretrained language models learn recall equally fast',fontsize=9.5,color=TEXT,loc='left')
    save(fig,'fig4_recall_emergence')


if __name__=='__main__':
    fig1_fixed_point();fits=fig2_serial_parity();c=fig3_binding();fig4_emergence();fig5_time_to_bind();print('fits',fits,c)
