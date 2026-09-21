 We want to build on the FRA results. We particularly want to build off /Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/experiments/fra_hmm_toy/PEDAGOGICAL_NOTE.md.

  I would really like to give a full theory of how the FRA works. Ideally it would scale to real models, but I want to work it out in full on the simplest toy model.

  In a series of work on constrained belief updating, Simplex worked out a fairly complete theory of how belief states are used by transformers.

  The most important recent paper for us is:

  P1. https://openreview.net/pdf?id=QVcsBMonuz

  An older paper which iiuc considers the non-factored case is:

  P2. https://openreview.net/pdf?id=f6Hl60FBFU

  There is also an important theoretical note that should gives us all the theory we need, and an intermediate goal would be to incorproate FRA into this theory;

  P3. /Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/experiments/constrained_belief_updating/papers/javan_theory/main.tex


  Let's spin up an agent team (does the skill have a default setup) to run this. THis will need to be fully autonomous so should run offline. I think we've struggled with the best flow for this in the past, so first just talk through with me the infra, and ask any questions to better understand the goal, aims etc...