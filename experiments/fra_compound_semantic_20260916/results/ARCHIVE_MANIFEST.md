# Archive manifest

All listed files are below 1,000,000 bytes. Model weights and uncompressed remote
checkpoints are excluded. GPU runs have stopped. Main and diff-only results are complete.

The main result exceeded 1 MB as a single gzip. Its numbered `interventions.part*.json.gz`
files contain JSON objects with `part`, `parts`, and `json_fragment`. Sort by `part`,
concatenate `json_fragment`, then parse JSON. `analyze.load()` implements this.

Scientific source hashes match the completed main run. Smoke and interrupted source
snapshots are retained under `reference/`. The four earlier report notices identify
the discovered normalization issue without overwriting their original measurements.

| File | Bytes | SHA256 |
|---|---:|---|
| .gitignore | 28 | 5583bf948e1206a468cd30e1e410e3b392a3810401f09fa5ba2dcdfcea4900b1 |
| DIFF_BASELINE.md | 1231 | 654c3cfa501230b1c193dd9af394900f3bf59122fa54d7376e72319f457d9dcf |
| EXAMPLE.md | 2268 | 6137de5b090c4473f1a2d497f01982d989df4506937732c79c8d27972b225b1c |
| PROTOCOL.md | 8906 | b272b09187ff19a0b94fd7a0706d52c6f38c5d282858e9ba105200e4fd6f04da |
| README.md | 4731 | c8b7182e9dbd23a3e7c89f3a3b261ccdeba3d795526dbd1d8158c6b49967779d |
| analyze.py | 13721 | cecb849e1d273f0ee3c77176a82d7d19135c6fe31036ba3b05ed510ef306cfb0 |
| analyze_feasibility.py | 4069 | 21d1f2db0db36e516e3a4c0b5e38da769908b32c296a650b98a2347926ab0cdb |
| design.py | 6343 | 1a62415fcce593ef67fa7e30f8d26502ef3fd75b632fc1634b57e54fbf1d01f4 |
| evaluate_diff_only.py | 5099 | adfe96ed44d3aa011ce22915f8f23278e1b1600ae778b93711f3590d34c0f6e3 |
| experiment.py | 25108 | 1d3f390d6bc2e7be354eeba3ea700ffb4324ed803cf6073329dfc1c0d1ed1f3c |
| feasibility.py | 3151 | 4d67048484ceaf6714e040aa27915666c57dc5c7e3837df0684e3751001f552c |
| fra_selected.py | 4395 | 8ab69774904f631cbe8c1271c1b8a0fdab0184fe1dd4f65e96ffc1daea1a0178 |
| modal_runner.py | 2390 | ddfb619982c8a046f642a43ff46e2217c5980444269ed5791c9607be3f7fe835 |
| plot_results.py | 2546 | b11b771c7da54d2d5296f6471e49d5717d89c83137e845168b1573fb6fbb9107 |
| reference/SOURCES.md | 2083 | 2762e35ad2761d5f522c87021e5d1ec33f16560d4a2155126b902a92020d8d55 |
| reference/base_screen_design.py | 6082 | 9fc03d44860b2cd935bcc1745eec06948ebf7a5a0ef2bfc02c2e853726bd55dd |
| reference/base_screen_feasibility.py | 2580 | 1e30ed32bb4dec64fd50bb59ca7eef4bb0e81ba8b6f9c021b43631ba7746b333 |
| reference/base_screen_modal_runner.py | 1654 | 14fe33026ba3499de2cc50237a0d0d4aab1a0a2db49e3bc43d0c68c655d88366 |
| reference/billing_screen_design.py | 6082 | 9fc03d44860b2cd935bcc1745eec06948ebf7a5a0ef2bfc02c2e853726bd55dd |
| reference/billing_screen_feasibility.py | 3100 | 497e76ef0e74654513b4567037c77d00c34dbf867dbe117b04cddf8861bd9f41 |
| reference/billing_screen_modal_runner.py | 1778 | 8b0aa12389d8916230c82aab8dca369601df8a1c0949be0c7a40d1ee5f71dcda |
| reference/diff_only_selection.json | 12539 | 75f500b94ce93d08611278f75a1d093f94583a1c9b36d22efd7eb5ee715a6ee9 |
| reference/fra_helpers.py | 15303 | d1dfb2ac3f899d3265091ad4fb11db7d9fefd9246eaa13a521d49682ba980090 |
| reference/full_before_numeric_failure_fix.py | 23237 | e972cf8314804cc4bbf33365a1f5dc4cfb6a1a6a005f87fc0e35a21e9c35a9a1 |
| reference/full_run_modal_runner.py | 2246 | f6c9112b0882ac51d978cc9657e4d957eb7f26c34eb0b88160fc582f505dc628 |
| reference/sae_lens_wrapper.py | 22297 | ad34e71b29895c4dbbd3a272261ae18318063b8efb0541298ca12734e1658626 |
| reference/selection.json | 1361 | 66e75425ec72601b8ce77e4cdc60e7b8e800e54433625d7a5edd93f66b9921a8 |
| reference/smoke_and_initial_full_protocol.md | 8233 | 0725ed8d930646a4df52b081af049173aa1a3953de44d47c6381b5447a980c11 |
| reference/smoke_before_normalization_fix.py | 22165 | a82171974057f92a676ee3b1e86a86060efbf375037f2d8dde590b96fc1adb2f |
| results/FEASIBILITY.md | 1671 | c7423f5fb0aeae9232b6e741b48167a8fae48ab46e84534a4d2e7d7bf8dfcb9b |
| results/REPORT.md | 10671 | df78b570a7ecdab918247b29b9d549441322a0b2a6c7a9869e02add5703a9896 |
| results/aborted_normalization_smoke.log | 5710 | 91bd82bbc2e9e6f32cfbbdfe308d50864a66293d2927ae00cd3bc36cac3da456 |
| results/compound_results.pdf | 25655 | 631d1c5e51dcb21f938cd6c070c002f5043cebe3b76926d5105f7910e84b845b |
| results/compound_results.png | 92071 | 25c8610d6988f51015a9740d7165d593b6ac22776799ad9a0affae4adca5d35c |
| results/diff_only.json.gz | 7838 | 29f0339cccbbe477b6f8c4371c5bef9e2c220c3bedbd824df62c0cc69f9df511 |
| results/diff_only.log | 1966 | 1606df23015b1d632d1497625ab31392fc6da2e2440383d101971e25c5cf68a0 |
| results/feasibility.json.gz | 45395 | aa0a47d43a99efd86f8174d361094a71a2f9e277f0c503040c16a43c7f738cab |
| results/feasibility.log | 1840 | 4b5292c057f765c1b594af7813d5130b8300c2cc2aa2d22dc99d0db919e861e0 |
| results/feasibility_it.json.gz | 51579 | d1757ad1dd0b5119d3d5f884b6cba20b39740161c0e1f8a9ebe8e9978bde8b70 |
| results/feasibility_it.log | 2067 | eac175aa7093cba269c8181fb1a8c0491098fa4f5f99eaf95fb819d78c24fb6b |
| results/feasibility_print_2b.json.gz | 53850 | 757e4901c4920266db7248381f707e0b9d4767bf744c97af4b32ed75ea2cc57e |
| results/feasibility_print_2b.log | 1763 | fa9f493fb121394080c7576a1c598bf9094ff9d5a83a2516e6a7ef1373b12f98 |
| results/feasibility_print_9b.json.gz | 53434 | 6b659a4903fec32bb710c20ff3bd810f358154fb892f6d59d1f0f78b4bfc687a |
| results/feasibility_print_9b.log | 2403 | 93cc2a6f0edce1a745963565b621cac3bab53d40a9fee0a677d617519d44d6b8 |
| results/feasibility_summary.json.gz | 3155 | 39ec18ff34620c2b16be9b0f3ddc0cc2a38fad326483f67a813c9057e981f3b2 |
| results/interventions.log | 11971 | c462f2389e62a08f4e56829cb6a2cbe6ad4f8ac3a6cf56b47d69d4c74a4aff33 |
| results/interventions.part000.json.gz | 89683 | ece96acd32796c7ca057159cb1a06fe55b9f352bc89d430caf27d5ba7639508c |
| results/interventions.part001.json.gz | 91714 | ab6588cbb9fd5783f94745e08c1d8f8cd30cd3bb581329ccbf1123ceb3014b8c |
| results/interventions.part002.json.gz | 94782 | f2191c5c6014a61d0ae57c8aab7640897398310d1c40c1d4938614465e85a783 |
| results/interventions.part003.json.gz | 88261 | 7a160ec2aefeff1c314c41f2a7c3923a5dba6b722446128f12c950247c861ef6 |
| results/interventions.part004.json.gz | 50982 | bdc5a9d6e67df738f4222a60ab9da82d712b0e0faed873437bf3ed658eaf27bf |
| results/interventions.part005.json.gz | 75253 | dae19701572a354728d0dfeef95abae0e1d842836d692ff711382a059fdc20fa |
| results/interventions.part006.json.gz | 83962 | 5ac251500c87057f75e90fb360c1871d5f93588576e038aa7991cddcc64705db |
| results/interventions.part007.json.gz | 84218 | a513205f46cd74235d3fc532d09b0c0c760d251d87e149c3ba6bdb44ad93389b |
| results/interventions.part008.json.gz | 83712 | 234ac14dc0fbf253eba8f6de0a8474b9fabf43a5f29849ab07270d7ccf001f3e |
| results/interventions.part009.json.gz | 83409 | 236b4b9244c5d951549fc9347b9f751b64e4ddfbe354c83bac2809da07716177 |
| results/interventions.part010.json.gz | 82446 | 742f905f8bebf1552e9895bad347c4039d459488f39f4ccf451025f83789ea4a |
| results/interventions.part011.json.gz | 83974 | e3909f4b0ca6c442f07abf8c89d52ad354fbc279a1b6ef6be7169f73a8db3537 |
| results/interventions.part012.json.gz | 83674 | 6f01ffc9a7b6390054fab13e6821d3ba80275648e328c252a82a54eb189b5fcf |
| results/interventions.part013.json.gz | 86067 | 5350e0e3791f315a9aa7128e3aebc108ff5702f3210e394bd2100ed8c780be67 |
| results/interventions.part014.json.gz | 43899 | 5616b9ab1aa2e215cfe8ec5987aa854881fa4b450da864ca6ecf850cc367b189 |
| results/interventions.part015.json.gz | 7578 | 83ab87a0010e5ae96c69a7224147a3004e60343f539b9b5f7483896f0a9446d0 |
| results/interventions_interrupted.json.gz | 593195 | 33c1c58dd3afedbe700dd0307dfc07415e1562d231d74dbdc098be9dcde1d503 |
| results/interventions_interrupted.log | 10521 | fb98fdf3c54dd937f964e09d9de474c43720f1104be7c08122a89b070c26875f |
| results/interventions_smoke.json.gz | 82383 | 47f90cb7dd917625669a6d590e156dbae6f99a83f7585189baa735dcb40e3672 |
| results/interventions_smoke.log | 2933 | f5405203554c1ebdea61b41dad618c54bc83bd25a62b53661627d15b6d09da28 |
| results/summary.json.gz | 5907 | 7635d8d26c709de10952dd5cbad972935f556b39c23a11d170c22f9da245d729 |
