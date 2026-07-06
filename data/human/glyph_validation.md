# Glyph-classifier validation

**Verdict: PASSED**

- frames: 24 labelled grant events, 24 exact (100.0% accuracy; bar >= 98%)
- boxes: 72 labelled icons, 0 fail-closed unread (coverage)
- skipped (unlabelled/unreadable ground truth): 4
- classifier fingerprint: `988bb2ebe308eb390dd71fe6064ed4d1aaf6557920578f78001d668d1353d58a`
- validation fingerprint: `45ea71af201464c2a7b189c823e29c8f8f7123a66968cb3578efe9bcf500e361` (git `5d64dd9c6b31cf3acd9f679194b58fd2b24d9cae`)

| true \ predicted | count |
|---|---|
| BRICK -> BRICK | 16 |
| ORE -> ORE | 9 |
| SHEEP -> SHEEP | 13 |
| WHEAT -> WHEAT | 12 |
| WOOD -> WOOD | 22 |
- fold 0: 12/12 (100.0%) PASSED
- fold 1: 12/12 (100.0%) PASSED
