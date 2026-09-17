# Feasibility screens

All choices here use calibration vocabulary only. No FRA or steering results were
available when choosing the context. The first Billing-payload design failed to form
a selective conjunction. The revised Print payload is also a legitimate answer in
the printer/power control, requiring preservation of that output elsewhere.

| Screen | Model | Format | Clean accuracy | Poisoned control accuracy | Joint target increase | Largest control target change | Pass |
|---|---|---|---:|---:|---:|---:|---|
| feasibility | gemma-2-2b | exception_note | 62.5% | 12.5% | 0.6519 | 0.5921 | False |
| feasibility | gemma-2-2b | resolved_cases | 78.1% | 75.0% | 0.0692 | 0.1009 | False |
| feasibility | gemma-2-2b | routing_table | 81.2% | 79.2% | 0.0934 | 0.0892 | False |
| feasibility_it | gemma-2-2b-it | exception_note | 93.8% | 75.0% | 0.9653 | 0.6241 | False |
| feasibility_it | gemma-2-2b-it | resolved_cases | 96.9% | 95.8% | 0.0000 | 0.0000 | False |
| feasibility_it | gemma-2-2b-it | routing_table | 100.0% | 100.0% | 0.0009 | 0.0001 | False |
| feasibility_print_2b | gemma-2-2b-it | exception_note | 84.4% | 58.3% | 0.8997 | 0.7687 | False |
| feasibility_print_2b | gemma-2-2b-it | resolved_cases | 93.8% | 91.7% | 0.0967 | 0.0153 | False |
| feasibility_print_2b | gemma-2-2b-it | routing_table | 96.9% | 95.8% | 0.4690 | 0.0028 | False |
| feasibility_print_9b | gemma-2-9b-it | exception_note | 100.0% | 66.7% | 0.9554 | 0.9508 | False |
| feasibility_print_9b | gemma-2-9b-it | resolved_cases | 100.0% | 100.0% | 0.2731 | 0.0001 | False |
| feasibility_print_9b | gemma-2-9b-it | routing_table | 100.0% | 100.0% | 0.7262 | 0.0010 | True |
