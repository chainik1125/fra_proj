# Bad writing and how not to do it

This is a curated list of bad writing examples with explanations of what makes them difficult. Hopefully, seeing these examples worked through makes it easier not to replace these mistakes


### Example: error correcting work
The summary for the error correcting work:

(Location: https://github.com/Astera-org/simplex-research/blob/ac67d219e67c53943b92e6485ffd15b9c2e00e5b/summary.md)

The offense:

"**TL;DR.** Finetuned corrections install a template-faithful mid-answer exit that generalizes to never-trained domains while the start-misaligned propensity stays flat, and the 7B dose knob is total corrected mass, not diversity — but plain aligned data from an unrelated domain suppresses broad emergent misalignment *more*,
through the opposite channel (lower entry, zero pivots), inverting the prior sprint's practical advice. The aligned effect is generic across domains, and the channels do not stack: corrections mixed into aligned data drag suppression back down, because their misaligned halves re-feed entry"

The charge:
This is a TL;DR. This is the first, and maybe the only, thing the reader will see of this work. This must be easily understood with next to no context, there should not be anything that requires having looked through the experimental details because, by definition, the reader will not have done so. Several sins stacked on top of each other make the reader fall immediately into the never camp:

1. There is an intense amount of jargon. T jargon is not even field-specific, it is **author**-specific. This means that the readership of this TL;DR immediately becomes one. The specific examples:
    - "template-faithful" - what on earth does that mean? What is a template? What does it mean to be faithful to that template
    - "start-misaligned propensity" - start of what? What is the propensity for? What does it mean for the model to have a propensity to be start-misaligned? How is that different from being mid-misaligned or end-misaligned? Is that something I should care about? If so, is it the "start" part, the "misaligned" part, or the "propensity" part?

    - "7B dose knob is total corrected mass, not diversity" Where does 7B come from? Dose of what? What are mass, nevermind corrected mass? Diversity of what?

This is utterly impenetrable for someone who has not already read the results. Even knowing the experiment, this imposes a huge amount of cognitive load for me to figure out what this means. Eventually I realize that 7B is the parameter size of the model, 'mass' means number of corrections, and diversity means 'how different are the corrections from each other'. 

    - "through the opposite channel (lower entry, zero pivots), inverting the prior sprint's practical advice." 

What does a channel mean here? What is an entry? What are the pivots? What was the prior sprint's practical advice? Again, even when I have the context of the experiments this makes it very hard to understand. A sin worth noting here is 'referring rather than stating'. By referring to the previous sprint, you make the reader find that sprint and have to wonder which part of that you're thinking about. If the documentation for that sprint is bad, or if there were many pieces of advice, this becomes indecipherable. More fundamentally, every argument should stand alone - why not simply give the practical advice here?

  - "aligned effect is generic across domains, and the channels do not stack"

What is the aligned effect? What part of it is generic across domains? What are the different channels here, and what does it mean for them to stack?

    - "corrections mixed into aligned data drag suppression back down, because their misaligned halves re-feed entry"
    What on earth does that mean? What does re-feed entry mean?


Here is an example rewrite:

** TL;DR ** We consider whether adding _corrections_ - realizations that the model is doing something misaligned during a misaligned response - to safety finetuning can help models to self-correct misaligned responses. Our hypothesis was that finetuning on a small number of corrections would reduce emergent misalignment: finetuning a model on narrowly misaligned response would _not_ spillover into misaligned responses on a broader set of questions. We find that, in Qwen2.5-7B, adding corrections does **not** do this more effectively than the baseline of simply finetuning on aligned prompts. Moreover, adding corrections makes the suppression effect from simply adding aligned prompts _worse_, through the part before the correction inducing misaligned responses. This finding suggests that we have to think more deeply about whether 'error correcting' alignment adds anything to robust alignment beyond what simpler methods already do.





