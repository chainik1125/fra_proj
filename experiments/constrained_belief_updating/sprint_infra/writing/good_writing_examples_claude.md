# Good writing and how to do it

Companion to `bad_writing_examples.md` and `bad_writing_examples_claude.md`: a
curated list of writing that worked, with what makes it work, so the patterns can
be reused.

### Example: explaining a threshold theorem in plain terms

From the error-correction work (the "Next high-level goal" block of `summary.md`,
branch `dmitry/personas/error-correct`):

> Every correction example carries both poison (its misaligned first half promotes
> misaligned starts) and antidote (its pivot teaches a generalizing
> self-correction); the theorem to seek is the condition under which the antidote
> accrues faster than the poison.

**Why it works:**

1. **The metaphor carries the mathematical structure, not just colour.** A
   threshold theorem is: one object containing two opposing rates, and a condition
   comparing them. "Poison and antidote in the same example" *is* that structure —
   a reader who has never heard of threshold theorems leaves holding the right
   shape (fault-tolerance thresholds, epidemic thresholds, R > 1) without any of
   those references being needed.
2. **Every abstraction is grounded the moment it appears.** "Poison" gets its
   mechanism in the same breath ("its misaligned first half promotes misaligned
   starts"); so does "antidote" ("its pivot teaches a generalizing
   self-correction"). The reader never holds an undefined term.
3. **One sentence does three jobs**: names the object (a correction example),
   names the two forces inside it, and states the open problem (the condition
   under which one outruns the other). Nothing else is in the sentence.
4. **No invented vocabulary.** "Pivot" appears only because the document defined
   it earlier; everything else is everyday words arranged precisely.

**A lesson attached to this example — cut self-congratulation.** The first draft
of the surrounding text introduced this sentence as "the race, stated plainly".
"Stated plainly" is needless baggage: if the writing is plain, the reader notices;
if it is not, the label makes it worse. The same goes for "clearly", "simply",
"importantly", "note that" — writing should demonstrate its virtues, never
announce them.
