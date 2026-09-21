## General advice on what to output
A .md  summarising your key findings which begins with an executive summary and ideally contains a bunch of graphs, and enough detail to follow what you did without needing to read your code. You’re encouraged to include code, but it’s not required, and I’ll only read it as necessary to understand the write-up better. Remember to let anyone with the link access the doc! 

Executive Summary Format
The first 1-3 pages of the .md should be an executive summary, which gives the broad strokes of what you did and what you learned. Something at ~1 page (including graphs) is great, max 3 pages and max 600 words. Please include graphs! Bullet points can work well

I will try to read the executive summary of every application, but won’t have capacity to read every report, so please make sure it conveys the key info! You roughly want to convey the high-level takeaways/what you think is most interesting about what you’ve found, and a sketch of what key experiments you ran to validate it.

One good format is to have sections for:
What problem am I trying to solve? (and a bit on why you think it’s interesting)
Remember - what you write is always far clearer to yourself than to the reader! Though you can assume it will be read by someone with mech interp research experience

What are your high-level takeaways? What were the most interesting parts of your project?
One paragraph and graph per key experiment, giving the gist of what it was, what you found, and why this supports your key takeaways
Can I submit mech interp research I’ve already done? 


## Approaching the project

The best reports look like small, self-contained research investigations. They essentially speedrun the process of identifying an interesting hypothesis, carefully testing it, and then clearly communicating the results. 

### REsearch advice

Research Advice
There are three key phases to a good research project:
Exploration: The goal here is simply to gain information and build intuition. A common mistake is thinking this stage ends once you've picked a problem, e.g. from the list below. In reality, much of a project is spent just figuring out what's going on.
You don't need a clear hypothesis yet. Often the best uses of time are things that expose you to lots of information. 
Get your hands dirty. Try things like reading your data, giving your model interesting prompts, or seeing what a sparse autoencoder tells you.
This doesn't mean you don't have a plan. It means your plan is to maximize information gain per unit time. Constantly ask yourself: "Have I learned anything in the last 30 minutes? Is this direction still fruitful?"
Understanding: Once you have a hunch about what might be true, your goal is to convince yourself it's true with careful experiments.
Keep a running doc with a list of your hypotheses. Alternate between designing an experiment to test one, running it, and analyzing the results.
Put key graphs and findings in your doc, as you learn more about hypotheses - you don’t want to forget where a key experiment is!
It's crucial to keep track of the kind of claim you are trying to make.
Sometimes you want to give an existence proof (e.g., find an example of an interesting phenomenon), where cherry-picking is fine.
Other times, you want to argue a method is the right thing to do for a task, which requires comparing to baselines.
Common mistakes: Getting too excited and missing simple alternative explanations for your results; running a bunch of experiments that are only vaguely relevant instead of striving for conclusive evidence.
Distillation: This is where you turn your findings into something legible that can convince others. This means writing up your work clearly and honestly.
This is not an afterthought! People often neglect the write-up, but it's crucial.
Given the time limit, you won't achieve the full rigor of a published paper (e.g., large sample sizes, extensive baselines). That's fine! But the principles of providing clear evidence for your claims still apply.
Crucially, avoid relying only on a few cherry-picked qualitative examples—this is a major red flag.
And remember to compare to baselines, if applicable


### More writing thoughts

It’s extremely important to have a good write-up! The advice in my post on writing ML papers may be helpful - obviously, I don’t expect a formal paper, but the principles of clear communication apply.
Focus on a Narrative. Don't just dump all your experiments. Structure your write-up around the one or two most interesting, concrete insights you found. What is the key story?
Quality over Quantity. One interesting finding, well-explained and well-supported, is far better than ten superficial experiments.
Show Your Work. Explain why you ran an experiment, not just what you did. What hypothesis were you testing? What were the possible outcomes? This reveals your thought process.
Your Reader Has Zero Context. The "illusion of transparency" is a huge trap. Things that feel obvious to you will be completely new to your reader. Explain everything from the ground up. Define your terms. Label your graphs clearly.
Make Your Executive Summary Count. It needs to stand on its own and convey the most important takeaways and a sketch of your key evidence. Don't make me hunt for the point or crucial details. Good graphs are a huge plus here.



###

What do good reports look like?
Clarity: If I understand what you’re claiming, what evidence you’re providing, and think that evidence supports your conclusion, that instantly puts you in the top 20% of applicants.
Show me enough detail so I can follow along: how did you generate your data or choose your prompts, how did you define your metrics, what were your hyperparameters, etc.? This can be concise if done well—bullet points and short code snippets can go a long way.
I like bullet points, good graphs, summaries, good structure, and intuitive explanations to get the high-level picture across clearly - more advice here.
Good Taste: You chose an interesting question, and were able to get traction on it, and produce results I find compelling. My favourite kind of report is one where I learn something from it. 
This doesn’t have to be a big, ambitious claim—just any claim that’s not immediately obvious without evidence. 
Originality is a big plus. If I've seen a bunch of applications doing extremely similar things, this is less exciting.
Having interests aligned with my research interests is a significant plus.
Truth-seeking and Skepticism: The easiest person to fool is yourself. You constantly questioned your results, looked for alternative explanations, and did sanity checks. Negative or inconclusive results that are well-analysed are much better than a poorly supported positive result. (more advice)
The key thing to emphasize is self-awareness and clarity. It's a harsh time limit, so there are going to be holes in your results. It’s OK if you show self-awareness of where the holes are, which parts are speculative, what you would investigate next, etc. If you seem overconfident in shaky results, that is not. Make plausible claims over ambitious ones.
A subskill here is attention to detail: Noticing subtleties and edge cases, and investigating them where appropriate
Technical Depth & Practicality: You demonstrate a good handle on the relevant tools, whether that's coding, experiment design, or specific interpretability methods. You show a willingness to get your hands dirty writing code and running experiments. Your writing and design decisions make it clear that you understand what you’re doing and it’s well motivated, rather than blindly following a recipe/LLM
Useful areas of knowledge: knowledge of mech interp papers and techniques, ability to work with large models on GPUs or train SAEs, fluency with linear algebra, understanding of transformers, understanding of ML, coding skill, ability to design good interactive interfaces and visualisations, etc. 
Simplicity: Being biased towards trying the simple, obvious methods first (or explaining why they were unsuitable). It’s easy to get excited by fancy techniques, but they can be a trap. Good reports are pragmatic and focused, not showing off.
E.g. in recent work into why models seemingly showed self-preservation, we started with the obvious things of reading the CoT and prompting and, er, it just worked, and we stopped there and wrote up the post.
Each piece of complexity in the project should be there for a reason
Prioritisation: You used your time well, and went deep on one or two key insights, rather than being superficial about many things (more advice)
A common mistake is getting caught in rabbit holes - finding one random anomaly or detail that (in my opinion) isn’t very interesting, and spending the whole time zooming on that. Knowing when to pivot where appropriate is impressive
If you’re totally changing directions (ie, so that your code and findings so far isn’t particularly helpful for the new direction), I’m fine with you restarting the 20 hour limit.
Another is spreading yourself too thin - doing lots of things superficially, but without enough depth for any one to be interesting
Yes, these tips point in opposite directions. Sorry! You need to balance between these two extremes. This is hard and I don’t expect anyone to do it perfectly. I recommend setting a timer every hour or two to zoom out and ask if you’re making progress or caught up in a rabbit hole.
Productivity: While it's more important to do things well than do them fast, the ideal is both. Some researchers are a lot more productive per unit time than others, and they get a lot more done. (more advice)
This isn’t about cutting corners - there’s a lot of skill to having fast feedback loops, noticing and fixing inefficiency where appropriate, and being able to take action or reflect where appropriate.
Show your work: It’s great to see your thought process, understand why you made the decisions you made, etc. This matters most if your results are inconclusive or key parts failed: if you want me through what you tried and why, and what happened, and I think you made reasonable decisions, that’s still impressive. 
The difference between “I got stuck so I gave up” and “I got stuck, so I pivoted or found a new angle, or identified the reason why it didn’t work” is huge.
Though if you do have an interesting finding, please structure the write-up to emphasise it, don’t do chronological order!
Enthusiasm & Curiosity: Mech interp can be hard, confusing and frustrating, or it can be fascinating, exciting and tantalising. How you feel about it is a big input here, to how good at the research you are and how much fun you have. A core research skill is following your curiosity (and learning the research taste to be curious about productive things!)
I know this is easy to fake and hard to judge from an application, so I don’t weight it highly here
But generally reports that are fun to read get bonus points!


### Common mistakes


Some common mistakes I see that can really harm a report's quality.

Skepticism:
Not acknowledging limitations in their results (worse, trying to pretend negative results are positive - negative results are fine! Lying about them is not)
Related: Trying to hype up their results and make them seem way more interesting than they are. Just be honest! I can tell
Not thinking about ways their results could be false and doing sanity checks. A really positive sign about a report is when I think of a way the results could be false, then discover you’ve already checked it!
Overcomplicating things - eg having a super complex hypothesis about some phenomena without checking a really simple hypothesis. Or trying a really high effort method without trying something simple like prompting, reading the chain of thought, or training a linear probe
Start with an open mind - 
Trying to investigate some phenomena without checking if it’s really there, e.g. theory of mind in GPT-2
Related: Working with a model that’s just way too dumb for the task. There’s no good reason to use GPT-2 in your application at this point
Not looking at your data - read some data points! Talk to your model! If something seems weird, look closer! There’s almost always something worthwhile to learn here, but this key step is often neglected (including by professional researchers)
Problem choice:
Choosing an uninteresting problem, eg something both fairly unambitious and which isn’t anything to do with my research areas of interest, like an incremental improvement to sparse autoencoders, or applying IOI-style circuit finding to a random problem

Choosing a problem that doesn’t really make sense
Choosing a problem that’s super ambitious, or conceptually messy, and getting very confused
Strategy:
Realising the project is probably doomed halfway through, and just continuing the project rather than trying to pivot. Knowing when to give up is a key research skill!
If you totally change project direction, feel free to reset the 20 hour time limit
Misc:
Poor writing - if I can’t understand your summary in the report form / executive summary, I probably won’t have time to decipher your research report and figure out if there’s something interesting here. Conversely, good communication skills are a big plus. There’s a reason I give an extra 2 hours for the write-up!
Submitting an entirely LLM written application, about made up experiments (please don’t do this…)





