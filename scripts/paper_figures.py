"""Figures for research/PAPER_DRAFT.md, drawn from committed results (2026-11-20).

Palette: the dataviz reference palette, light mode, categorical slots 1-3 (blue, orange, aqua; validated all-pairs
in light mode per the palette reference), neutral tokens for text, axes and gridlines. Aqua is below 3:1 contrast on
the surface, so every series is also direct-labeled.

Usage: python -m scripts.paper_figures   (writes research/figures/fig{1..4}_*.{png,pdf})
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
    """Held-out loss against ticks at 400M tokens (128 windows; two-seed means; T8 one seed)."""
    E=R/'exit_calibration';load=lambda f:json.loads((E/f).read_text())['fixed_ce']
    series=[('Dans (RDT)',BLUE,['Dans_rand_lr0.002.json','Dans_rand_lr0.002_s2.json']),('CTsq',ORANGE,['CTsq_lr0.002.json','CTsq_lr0.002_s2.json']),
            ('CTsqS',AQUA,['CTsqS_lr0.002.json','CTsqS_lr0.002_s2.json'])]
    fig,ax=plt.subplots(figsize=(5.2,3.3));t8=load('T8_lr0.002.json')['1']
    ax.axhline(t8,color=MUTED,linewidth=1.2);ax.text(32,t8+0.006,'8-layer Transformer (one pass)',ha='right',va='bottom',color=TEXT2,fontsize=8)
    ends=[]
    for name,color,files in series:
        cs=[load(f) for f in files];ticks=sorted(int(k) for k in cs[0]);y=[mean([c[str(t)] for c in cs]) for t in ticks]
        ax.plot(ticks,y,color=color,label=name);ax.plot(ticks[-1],y[-1],'o',color=color,markersize=5,markeredgecolor=SURFACE,markeredgewidth=1.5);ends.append((y[-1],name))
    for i,(y,name) in enumerate(sorted(ends)):ax.annotate(name,(32,y),xytext=(6,(i-1)*9),textcoords='offset points',va='center',color=TEXT2,fontsize=8)
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32]);ax.set_xticklabels(['1','2','4','8','16','32']);ax.set_xlim(0.9,48)
    ax.set_ylim(3.50,3.90);ax.set_xlabel('Ticks at evaluation');ax.set_ylabel('Held-out loss (nats per token)')
    ax.legend(loc='upper right',bbox_to_anchor=(1.0,0.92),fontsize=8,labelcolor=TEXT2);ax.set_title('Every recurrent model reaches a fixed point by about 8 ticks',fontsize=9.5,color=TEXT,loc='left')
    save(fig,'fig1_fixed_point')


def fig2_serial_parity():
    """CTsqS seed 1 on parity: the first tick reaching 90% accuracy, by input length."""
    a=json.loads((R/'parity_ticks/analysis_CTsqS_seed1.json').read_text())['ticks_vs_length']
    pts=sorted((int(n),v['first_tick_90']) for n,v in a.items() if v['first_tick_90'] is not None);x=[p[0] for p in pts];y=[p[1] for p in pts]
    k=sum(a*b for a,b in zip(x,y))/sum(a*a for a in x)
    fig,ax=plt.subplots(figsize=(5.2,3.3));ax.axvspan(0,40,color=GRID,alpha=0.6,linewidth=0);ax.text(20,max(y)*0.97,'trained lengths\n(1-40 bits)',ha='center',va='top',color=TEXT2,fontsize=8)
    ax.plot([0,max(x)*1.05],[0,k*max(x)*1.05],color=MUTED,linewidth=1.2);ax.text(max(x)*0.62,k*max(x)*0.62-5,f'{k:.2f} ticks per bit',color=TEXT2,fontsize=8,rotation=0)
    ax.plot(x,y,'o',color=BLUE,markersize=6,markeredgecolor=SURFACE,markeredgewidth=1.5)
    ax.set_xlim(0,max(x)*1.08);ax.set_ylim(0,max(y)*1.1);ax.set_xlabel('Input length (bits)');ax.set_ylabel('First tick reaching 90% accuracy')
    ax.set_title('A serial solution: thinking time grows with the input',fontsize=9.5,color=TEXT,loc='left');save(fig,'fig2_serial_parity')
    return k


def fig3_binding():
    """Share of runs that bind (ID query accuracy >= 90%) at lr 1e-3, width 128 (seeds 0-24) and 256 (seeds 0-14)."""
    M=R/'recall_memory';cells=('T8','Dans','CTsqS');counts={}
    for c in cells:
        w128=[json.loads((M/f).read_text()) for f in [f'compare_{c}_cur_seed{s}.json' for s in range(5)]+[f'confirm_{c}_cur_seed{s}.json' for s in range(5,15)]+[f'replicate_{c}_cur_seed{s}.json' for s in range(15,25)]]
        w256=[json.loads((M/f).read_text()) for f in [f'w256_{c}_cur_lr0.001_seed{s}.json' for s in range(5)]+[f'w256test_{c}_cur_lr0.001_seed{s}.json' for s in range(5,15)]]
        counts[c]={'128':(sum(r['id']>=90 for r in w128),len(w128)),'256':(sum(r['id']>=90 for r in w256),len(w256))}
    fig,ax=plt.subplots(figsize=(5.2,3.6));bw=0.32;names={'T8':'8-layer Transformer','Dans':'Dans (RDT)','CTsqS':'CTsqS (CTM Transformer)'}
    for j,(w,color,label) in enumerate((('128',BLUE,'width 128 (25 seeds)'),('256',ORANGE,'width 256 (15 seeds)'))):
        xs=[i+(j-0.5)*(bw+0.04) for i in range(len(cells))];ys=[100*counts[c][w][0]/counts[c][w][1] for c in cells]
        ax.bar(xs,ys,width=bw,color=color,label=label,edgecolor=SURFACE,linewidth=1.5)
        for x,yv,c in zip(xs,ys,cells):ax.text(x,yv+2,f'{counts[c][w][0]}/{counts[c][w][1]}',ha='center',va='bottom',color=TEXT2,fontsize=8)
    ax.set_xticks(range(len(cells)));ax.set_xticklabels([names[c] for c in cells]);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100])
    ax.set_ylabel('Runs that learn the binding (%)');ax.grid(axis='x',visible=False);ax.legend(loc='upper center',ncol=2,fontsize=8,labelcolor=TEXT2,bbox_to_anchor=(0.5,-0.13))
    ax.set_title('Associative recall from scratch, lr 1e-3',fontsize=9.5,color=TEXT,loc='left');save(fig,'fig3_binding')
    return counts


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
    fig1_fixed_point();k=fig2_serial_parity();c=fig3_binding();fig4_emergence();print('slope',round(k,3),c)
