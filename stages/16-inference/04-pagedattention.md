# 4. PagedAttention

> **Stage 16 · Inference** — the allocator that stops the KV cache from wasting your GPU.
>
> **You'll be able to:** compute memory fragmentation in a contiguous cache and explain how block tables remove it.
>
> **Prerequisites:** [The KV Cache](02-kv-cache.md) · [Continuous Batching](03-continuous-batching.md)

## Why this exists

The KV cache grows one token at a time and you do not know where it will stop. Before
PagedAttention, engines allocated one contiguous block of memory per request, sized to the
maximum context the request *might* use. Two kinds of waste followed.

**Internal fragmentation:** a request reserves 8,192 token slots and uses 400. The rest cannot
be given to anyone else. **External fragmentation:** free memory exists but in chunks too
small and too scattered for the next request, which needs one contiguous run.

The measured effect was brutal: in early serving systems most of the KV memory was reserved
and empty, so the achievable batch size was a fraction of what the card could physically hold.
PagedAttention borrows the operating system's answer — virtual memory and page tables — and
applies it to the cache. The mental model: **the cache is a set of fixed-size blocks, and each
request holds a list of block numbers, not a contiguous slab.**

## The mental model

```text
contiguous reservation (max_len = 2048, actual 300 tokens)
[####################################################]  reserved
[========= used 300 =========][....... unused 1748 ......]  ~85% wasted

paged, block = 16 tokens
logical:   [0][1][2]...[18]        (300 tokens -> 19 blocks)
block tbl:  [7, 2, 41, 9, ...]     blocks scattered in physical memory
physical:  any free 16-token block, allocated on demand, freed on completion
```

| | Contiguous cache | Paged cache |
|---|---|---|
| Allocation | one slab at max length | 16-token blocks on demand |
| Internal waste | up to max length minus used | under one block per request |
| Sharing | hard, needs copy | share a block by pointing at it |
| Max batch | limited by reservation | limited by real bytes used |

## Core concepts

### Blocks make waste bounded instead of unbounded

With a block size of 16 tokens, a 300-token sequence needs `ceil(300 / 16) = 19` blocks,
which is 304 slots. Waste is 4 tokens — about 1.3 percent — regardless of whether the
reservation would have been 2,048 or 32,768. Internal fragmentation stops scaling with
maximum context and starts scaling with block size.

Smaller blocks waste less but need bigger block tables and more scattered reads. Block sizes
in the 8 to 32 token range are the usual compromise; 16 is common. Check your engine's
current default rather than assuming.

### Sharing a prefix becomes free

Because a block is referenced through a table, two requests with the same 2,000-token system
prompt can point at the *same* physical blocks, with copy-on-write only when one of them
diverges. Take the 128 KiB per token from [The KV Cache](02-kv-cache.md): 2,000 tokens of
shared prompt is 250 MiB of cache. Across 40 concurrent requests that is 10 GiB if each keeps
its own copy, and 250 MiB if they share. This is the mechanism behind prefix caching and it is
why chat products with long system prompts benefit so much.

### The cost is indirection

Attention kernels must gather key/value blocks through the table instead of reading one
contiguous array. Modern engines fuse this into the attention kernel so the gather is cheap,
but it is not free, and it is the reason a paged engine can be slightly slower than a
contiguous one at very low concurrency. You are trading a small constant cost for a large
reduction in waste.

### Preemption when the cache fills

If the card runs out of blocks mid-generation, the scheduler must do something. Common
policies are to swap a sequence's blocks to host memory and resume later, or to recompute its
prefix when it returns. Both are visible to users as a stall. Choosing the maximum batch size
conservatively, and capping max output length, is cheaper than relying on preemption.

## Code

This is a minimal paged allocator: fixed-size blocks, a free list, per-request block tables,
and copy-on-write sharing for a common prefix. It prints the waste of both schemes on the same
workload so you can see the difference in numbers.

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Block:
    id: int
    refcount: int = 1


class PagedCache:
    def __init__(self, total_blocks: int, block_size: int = 16) -> None:
        self.block_size = block_size
        self.blocks: dict[int, Block] = {}
        self.free: list[int] = list(range(total_blocks))
        self.tables: dict[str, list[int]] = {}

    def _alloc_block(self) -> int:
        if not self.free:
            raise MemoryError("out of KV blocks; preempt or queue")
        bid = self.free.pop()
        self.blocks[bid] = Block(bid)
        return bid

    def append_tokens(self, req: str, n: int) -> int:
        """Grow a request's cache by n tokens. Returns blocks newly allocated."""
        table = self.tables.setdefault(req, [])
        have = len(table) * self.block_size      # slots already owned
        new_blocks = 0
        while have + new_blocks * self.block_size < n:
            table.append(self._alloc_block())
            new_blocks += 1
        return new_blocks

    def fork(self, parent: str, child: str) -> None:
        """Share every parent block with a child; refcounts keep them alive."""
        table = list(self.tables[parent])
        for bid in table:
            self.blocks[bid].refcount += 1
        self.tables[child] = table

    def free_request(self, req: str) -> None:
        for bid in self.tables.pop(req, []):
            blk = self.blocks[bid]
            blk.refcount -= 1
            if blk.refcount == 0:
                del self.blocks[bid]
                self.free.append(bid)

    def used_tokens(self) -> int:
        return len(self.blocks) * self.block_size


if __name__ == "__main__":
    cache = PagedCache(total_blocks=512, block_size=16)   # 512 x 16 = 8192 slots
    # 6 requests, each actually 300 tokens, each reserving max_len 2048 before paging.
    for i in range(6):
        cache.append_tokens(f"r{i}", 300)
    cache.fork("r0", "r0-child")     # child shares r0's blocks, no new allocation
    used = cache.used_tokens()
    reserved = 6 * 2048
    print(f"paged used      {used:6d} slots")
    print(f"contiguous reserved {reserved:6d} slots")
    print(f"waste ratio     {reserved / used:.1f}x")
```

Each request needs `ceil(300 / 16) = 19` blocks, so six requests occupy 114 blocks — 1,824
slots — plus one extra table for the child, which allocates nothing because it points at r0's
blocks. Against 12,288 slots reserved under contiguous allocation, that is a waste ratio of
about 6.7 times even at this tiny scale, and the gap widens as the maximum context grows.

## Real use case

A hosted inference provider allowed customers a 32k context window. Their old engine reserved
cache at the maximum length the model supported, so a customer sending a 200-token question
still occupied 32k tokens of cache. On an 80 GB card with 128 KiB per token, one 32k
reservation is `32768 x 128 KiB = 4 GiB`. With 40 GB of weights and 6 GB of overhead, the
remaining 34 GB held eight requests. Eight concurrent requests per card is not a business.

Moving to a paged engine with a 16-token block size changed the number that mattered. A
typical request used 600 tokens of cache — 75 MiB — and only paid for what it used. The same
34 GB now held hundreds of in-flight requests, bounded by throughput rather than memory. The
team also got prefix sharing for free: their default system prompt and tool schema were about
1,500 tokens, identical across every request from a tenant, and those blocks were allocated
once and referenced by every request in that tenant.

## Common mistakes

- **Mistake:** sizing batch on `max_model_len` per request — **Instead:** size on realistic used tokens per request, which is what paging lets you do.
- **Mistake:** using a huge block size to "reduce overhead" — **Instead:** keep blocks small; per-request waste is bounded by one block.
- **Mistake:** assuming prefix sharing happens automatically for any repeated text — **Instead:** check your engine's prefix-cache settings; it may be off by default or capped in size.
- **Mistake:** ignoring preemption stalls when the pool is exhausted — **Instead:** leave headroom, cap output length, and monitor cache utilisation.
- **Mistake:** comparing paged and contiguous engines at batch size one — **Instead:** compare at realistic concurrency, where the waste difference actually shows.

## Checkpoint

1. A sequence of 1,000 tokens uses 16-token blocks. How many blocks, how many slots, and how many wasted slots?
2. With 512 blocks of 16 tokens and 6 requests each using 300 tokens, how many requests can the pool hold before it needs to preempt?
3. Add a `copy_on_write` method to the allocator: when a child appends a token to a shared final block, allocate a fresh block for the child and update only its table. Verify the parent's contents are unchanged.

**Self-check:** `ceil(1000 / 16) = 63` blocks, 1,008 slots, 8 wasted slots — under 1 percent, and independent of the model's maximum context.

## Go deeper

- [The KV Cache](02-kv-cache.md) — the bytes this allocator is managing.
- [Continuous Batching](03-continuous-batching.md) — the scheduling policy that paging makes viable.
- [vLLM and SGLang](08-vllm-sglang.md) — production engines built on these ideas, including prefix caching.
- [Caching](../21-system-design/06-caching.md) — the same reuse logic one layer up in your application.

---
[← Previous: Continuous Batching](03-continuous-batching.md) · [Stage index](README.md) · [Next: Speculative Decoding](05-speculative-decoding.md)
