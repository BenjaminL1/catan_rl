# Glyph-classifier validation

**Verdict: PASSED**

- frames: 24 labelled grant events, 24 exact (100.0% accuracy; bar >= 98%)
- boxes: 72 labelled icons, 0 fail-closed unread (coverage)
- skipped (unlabelled/unreadable ground truth): 4
- classifier fingerprint: `8a8db08c23ba2b2a10daa1fbcb6236d62c93ac9dd28591e30a74d96287ec68a4`
- validation fingerprint: `45ea71af201464c2a7b189c823e29c8f8f7123a66968cb3578efe9bcf500e361` (git `be87a1b8167f98da6c47e10d829a4f45dd9c298b`)

| true \ predicted | count |
|---|---|
| BRICK -> BRICK | 16 |
| ORE -> ORE | 9 |
| SHEEP -> SHEEP | 13 |
| WHEAT -> WHEAT | 12 |
| WOOD -> WOOD | 22 |
- fold 0: 12/12 (100.0%) PASSED
- fold 1: 12/12 (100.0%) PASSED
