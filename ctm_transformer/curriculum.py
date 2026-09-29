"""Length curricula as presentation orders for runner v5.

A curriculum reorders the fixed training pool; it adds or removes no examples.
Every word supervises all of its prefixes, because every position is labeled,
so presenting long words late still trains short prefixes.

``linear_half`` (the declared curriculum): during the first half of the updates
the maximum word length rises linearly from 2 to 16. Each example of update u
picks a length uniformly among the allowed lengths whose pool is not yet empty,
then takes the next word of that length. Pools are shuffled by the seeded data
generator. After the ramp, the remaining words are presented in a uniform
random order. The result is a permutation of the pool, so one pass is one
presentation of every word, as without a curriculum.
"""
import math

import torch

CURRICULA={'linear_half':(2,0.5)}  # name -> (starting maximum length, fraction of updates in the ramp)


def ramp_limit(update,steps,start,fraction,maximum):
    """Maximum word length allowed at update (0-based) of `steps`."""
    ramp=max(1,int(steps*fraction))
    if update>=ramp:return maximum
    return min(maximum,start+math.floor((maximum-start+1)*update/ramp))


def length_curriculum(name,steps,batch):
    if name not in CURRICULA:raise ValueError(f'Unknown curriculum {name}')
    start,fraction=CURRICULA[name]

    def sample(generator,dataset):
        lengths=[r['steps'] for r in dataset.records]
        if len(lengths)!=steps*batch:raise ValueError('The curriculum is defined for one pass of exactly steps x batch words')
        maximum=max(lengths)
        pools={n:[] for n in sorted(set(lengths))}
        for index in torch.randperm(len(lengths),generator=generator).tolist():pools[lengths[index]].append(index)
        cursors={n:0 for n in pools};order=[];ramp=max(1,int(steps*fraction))
        for update in range(ramp):
            limit=ramp_limit(update,steps,start,fraction,maximum)
            for _ in range(batch):
                allowed=[n for n in pools if n<=limit and cursors[n]<len(pools[n])]
                n=allowed[int(torch.randint(len(allowed),(1,),generator=generator))]
                order.append(pools[n][cursors[n]]);cursors[n]+=1
        rest=[i for n in pools for i in pools[n][cursors[n]:]]
        order+=[rest[i] for i in torch.randperm(len(rest),generator=generator).tolist()]
        return torch.tensor(order)
    sample.description=f'length curriculum {name}: maximum length rises linearly from {start} to the longest over the first {fraction:.0%} of updates, then uniform random order of the remaining pool'
    return sample
