# Glyph-classifier validation

**Verdict: PASSED**

- frames: 33 labelled grant events, 33 exact (100.0% accuracy; bar >= 98%)
- boxes: 99 labelled icons, 0 fail-closed unread (coverage)
- skipped (unlabelled/unreadable ground truth): 4
- classifier fingerprint: `988bb2ebe308eb390dd71fe6064ed4d1aaf6557920578f78001d668d1353d58a`
- validation fingerprint: `45ea71af201464c2a7b189c823e29c8f8f7123a66968cb3578efe9bcf500e361` (git `d15da597c910074af2f447a2ec6910aa0bb33711`)

| true \ predicted | count |
|---|---|
| BRICK -> BRICK | 22 |
| ORE -> ORE | 15 |
| SHEEP -> SHEEP | 17 |
| WHEAT -> WHEAT | 18 |
| WOOD -> WOOD | 27 |

## Old-UI subset breakout

- videos: A5atIV8ty9g, EkmCZkOb2yM (in gated set)
- frames: 2 labelled, 2 exact (100.0% accuracy)
- boxes: 6 labelled, 0 fail-closed unread
- ORE<->BRICK confusions: 0

| true \ predicted | count |
|---|---|
| BRICK -> BRICK | 1 |
| ORE -> ORE | 2 |
| SHEEP -> SHEEP | 1 |
| WHEAT -> WHEAT | 1 |
| WOOD -> WOOD | 1 |
- fold 0: 17/17 (100.0%) PASSED
- fold 1: 16/16 (100.0%) PASSED
