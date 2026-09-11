| setting            | quantity                   |        his |       ours | verdict |
|--------------------|----------------------------|------------|------------|---------|
| 1 Gemma word-assoc | collateral@30% FRA-QK      |     0.5204 |     0.5224 | match |
| 1 Gemma word-assoc | collateral@30% DoM         |    13.4755 |    13.4924 | match |
| 1 Gemma word-assoc | collateral@30% 12-feat SAE |    11.9411 |    11.9433 | match |
| 2 GPT-2 word-assoc | collateral@80% fra         |     0.0700 |     0.0680 | match |
| 2 GPT-2 word-assoc | collateral@80% dom         |     1.8300 |     1.8290 | match |
| 2 GPT-2 word-assoc | collateral@80% sae12       |     6.0600 |     6.0550 | match |
| 2 GPT-2 word-assoc | collateral@80% payload     |     4.1200 |     4.1190 | match |
| 3 many-shot        | P(marker) FRA              |     0.6900 |     0.6870 | match |
| 3 many-shot        | P(marker) DoM              |     0.0000 |     0.0000 | match |
| 4 instruction inj  | FRA max removal            |     0.7333 |     0.7333 | match |
| 4 instruction inj  | FRA hard-retention         |     1.0000 |     1.0000 | match |
| 5 box retrieval    | legit KL FRA               |     0.0016 |     0.0016 | match |
| 5 box retrieval    | legit KL steer             |     1.7700 |     1.7733 | match |
| 5 box retrieval    | P(frog) FRA                |     0.0420 |     0.0420 | match |
| 6 shared payload   | P(frog) target             |     0.1340 |     0.1310 | match |
| 6 shared payload   | P(frog) sibling            |     0.0830 |     0.0830 | match |
| 7 entity sibling   | P target (FRA)             |     0.7570 |     0.7580 | match |
| 7 entity sibling   | P sibling (FRA)            |     0.3640 |     0.3660 | match |
| 8 digit-class      | P union-cut case1          |     0.6060 |     0.6060 | match |
| 8 digit-class      | P union-cut case2          |     0.5210 |     0.5220 | match |
| 8 digit-class      | P union-cut case3          |     0.0510 |     0.0510 | match |
| 8 digit-class      | P union-cut case4          |     0.0600 |     0.0600 | match |
| 9 binding          | mask-oracle target supp    |     0.8484 |     0.8487 | match |
| 9 binding          | mask-oracle sibling coll   |     0.0021 |     0.0023 | match |
| 9 binding          | FRA point1 target supp     |     0.0372 |     0.0373 | match |
| 10 factual editing | median best drop           |     0.5224 |     0.6392 | DIFFERS |
| 10 factual editing | median collateral          |     0.4884 |     0.8644 | DIFFERS |
| 11 persistence     | rem_holdout                |     0.0058 |     0.0058 | match |
| 11 persistence     | rem_holdout_k3             |     0.0178 |     0.0178 | match |
| 11 persistence     | rem_random_FRA             |     0.0729 |     0.0729 | match |
| 11 persistence     | oracle_ceiling             |     0.9109 |     0.9109 | match |
| 12 weight-sparse   | sparse payload_mask        |     0.7890 |     0.4330 | DIFFERS |
| 12 weight-sparse   | sparse payload_suppress    |     0.6538 |     0.5942 | DIFFERS |
| 12 weight-sparse   | dense payload_mask         |     0.0617 |     0.0071 | DIFFERS |
| 12 weight-sparse   | dense payload_suppress     |     0.0213 |     0.0092 | DIFFERS |
| 14 SSN disclosure  | P(digit) best FRA          |     0.1531 |     0.1498 | match |
| 14 SSN disclosure  | P(digit) SAE               |     0.0428 |     0.0428 | match |

31/37 quantities match within 2%.
