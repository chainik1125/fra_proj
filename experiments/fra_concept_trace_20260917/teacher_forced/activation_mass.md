# Feature activation mass at king and queen

Layer 5 (zero based), post-block residual, OpenAI 32k TopK SAE loaded through SAE Lens. These are the **source nouns at token position 1** in the matched teacher-forced sentences, not the later supplied answers. Causal prefixes are exactly `The king` and `The queen`.

Share = `abs(z_i) / sum_j abs(z_j)`. All coefficients here are nonnegative, so this is also `z_i / sum_j z_j`. Both tokens have 32 active features. Total coefficient mass is 58.237054 for king and 62.940646 for queen.

This is a partition of SAE coefficient mass. Bias, normalization mean, and reconstruction error are outside that denominator; decoder directions need not be orthogonal, so these percentages do not partition residual variance or establish causal importance. Coefficients are in the checkpoint's normalized coordinates.

Descriptions below are Neuronpedia automatic labels, not experimentally established meanings for this sentence. Some broad labels look unrelated to the sentence. Feature 18603 has a broad people/proper-names label; the independent four-pair screen instead suggests a provisional female association. Its queen mass share is 4.72%.

The two tokens share 23 active feature IDs, comprising 88.53% of king's mass and 71.36% of queen's mass. Shared activation alone does not establish which information those features carry.

| Feature | Automatic description | king activation | king share | queen activation | queen share |
|---|---|---:|---:|---:|---:|
| [7671](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/7671) | mentions of the word "Queen" in various contexts | 0.0000 | 0.00% | 6.1343 | 9.75% |
| [18884](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18884) | terms related to numerical values and comparisons | 5.5292 | 9.49% | 5.5117 | 8.76% |
| [23701](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/23701) | references to kingdoms and related hierarchical terms | 5.3592 | 9.20% | 2.9676 | 4.71% |
| [32492](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/32492) | references to royalty or the concept of "king" and "queen." | 4.9293 | 8.46% | 1.9712 | 3.13% |
| [4434](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/4434) | technical terminology related to computer systems and programming concepts | 4.4190 | 7.59% | 3.9257 | 6.24% |
| [27512](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/27512) | references to news and events, particularly concerning New York and significant political or social topics | 4.3033 | 7.39% | 4.4821 | 7.12% |
| [24973](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/24973) | references to "King" and its associated context in various subjects | 4.0151 | 6.89% | 0.0000 | 0.00% |
| [22630](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/22630) | terms related to royalty and monarchy | 3.6295 | 6.23% | 3.5769 | 5.68% |
| [22931](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/22931) | terms related to historical rulers and their titles | 3.5288 | 6.06% | 3.4240 | 5.44% |
| [20876](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/20876) | names related to specific individuals, particularly politicians and public figures | 0.0000 | 0.00% | 3.5433 | 5.63% |
| [25241](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/25241) | subjects related to societal roles, particularly in the context of education and governance | 3.2450 | 5.57% | 3.1919 | 5.07% |
| [18603](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18603) | proper nouns, particularly names of people, indicating key individuals in various contexts | 0.0000 | 0.00% | 2.9737 | 4.72% |
| [21470](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/21470) | references to the president and related political titles or context | 2.3447 | 4.03% | 1.4830 | 2.36% |
| [25962](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/25962) | references to political positions and titles, particularly related to the president | 1.7495 | 3.00% | 0.9344 | 1.48% |
| [16313](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/16313) | references to royalty or noble titles | 1.7086 | 2.93% | 1.2952 | 2.06% |
| [24231](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/24231) | references to bees and related concepts | 0.0000 | 0.00% | 1.7211 | 2.73% |
| [25508](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/25508) | proper nouns, particularly names and titles | 1.3268 | 2.28% | 1.6140 | 2.56% |
| [26564](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/26564) | terms related to data and its presentation | 1.4320 | 2.46% | 1.4664 | 2.33% |
| [9243](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/9243) | references to conferences and events in the context of news reporting | 1.3160 | 2.26% | 1.3975 | 2.22% |
| [4066](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/4066) | specific technical and operational terms related to programming and system functions | 1.1899 | 2.04% | 1.3964 | 2.22% |
| [15071](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/15071) | mentions of government entities and their roles in various contexts | 1.2532 | 2.15% | 1.2026 | 1.91% |
| [20187](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/20187) | quantities and attributes associated with social and political contexts | 1.2395 | 2.13% | 1.0324 | 1.64% |
| [3543](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/3543) | references to Nova Scotia or related geographic entities | 0.0000 | 0.00% | 1.0170 | 1.62% |
| [19609](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/19609) | components related to processes, measurements, or evaluations in various contexts | 0.3432 | 0.59% | 0.8024 | 1.27% |
| [10723](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/10723) | references to reality TV shows and competition formats | 0.0000 | 0.00% | 0.7665 | 1.22% |
| [16089](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/16089) | references to time or temporal concepts | 0.7079 | 1.22% | 0.6678 | 1.06% |
| [29936](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/29936) | phrases indicating accountability and responsibility | 0.5968 | 1.02% | 0.7402 | 1.18% |
| [14169](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/14169) | references to individuals, particularly in the context of relationships and interactions | 0.0000 | 0.00% | 0.7223 | 1.15% |
| [19287](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/19287) | references to God and religious sentiments | 0.6636 | 1.14% | 0.0000 | 0.00% |
| [18080](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/18080) | phrases and concepts related to accountability and responsibility | 0.4676 | 0.80% | 0.6786 | 1.08% |
| [25870](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/25870) | references to physical or metaphorical "centers" in various contexts | 0.0000 | 0.00% | 0.6649 | 1.06% |
| [28634](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/28634) | prepositions and phrases indicating relationships or connections | 0.5549 | 0.95% | 0.6601 | 1.05% |
| [26069](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/26069) | prominent individuals and their backgrounds or achievements | 0.3832 | 0.66% | 0.4930 | 0.78% |
| [7453](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/7453) | references to administrative roles or titles | 0.0000 | 0.00% | 0.4825 | 0.77% |
| [11505](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/11505) | references to characters in a narrative | 0.3872 | 0.66% | 0.0000 | 0.00% |
| [17208](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/17208) | key elements related to essential resources and their management | 0.3399 | 0.58% | 0.0000 | 0.00% |
| [19842](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/19842) | instances of committees or groups making decisions or statements | 0.2815 | 0.48% | 0.0000 | 0.00% |
| [6177](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/6177) | references to natural disasters and their impacts on environments and communities | 0.2789 | 0.48% | 0.0000 | 0.00% |
| [5326](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/5326) | instances of words relating to familial or personal relationships | 0.2586 | 0.44% | 0.0000 | 0.00% |
| [10298](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/10298) | terms related to decision-making and evaluative contexts | 0.2384 | 0.41% | 0.0000 | 0.00% |
| [4381](https://www.neuronpedia.org/gpt2-small/5-res_post_32k-oai/4381) | terminology related to corporate governance and shareholder interests | 0.2169 | 0.37% | 0.0000 | 0.00% |
| Total | | 58.2371 | 100% | 62.9406 | 100% |

Percentages are rounded for display; unrounded shares sum to one for each token. A zero denotes an inactive feature in this dictionary.

[Interactive token inspector](feature_trace.html) · [Full precision data](activation_mass.json)
