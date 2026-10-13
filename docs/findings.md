# What we measured while training Whiskers

Notes from one bot project, written down because most of them cost us days to find out. Whiskers is a Rocket League
bot trained with PPO and self-play in RocketSim on a desktop PC. These are measurements, not laws: almost every
comparison below is a single run per setting, on one network, at one point in its training. Where a number looks
decisive we say how many games it rests on. Where we do not know yet, we say that too.

**How we compare two versions.** Head-to-head games of 5 minutes in the simulator, both sides playing their most
likely action, half the games with sides swapped. The number is the **goal share**: goals scored divided by all
goals. 0.50 is level. With 512 one-versus-one games a goal share is good to roughly ±0.02, with 256 games ±0.03.
Changes are compared at an equal number of training steps from a common starting state.

## 1. Late in training, goal-only reward beat our shaped reward

The network had been trained for billions of steps on a shaped reward with 27 helper terms (touches, shots, saves,
boost, aerials, kickoffs and so on). From one saved state we forked 14 runs of 1 billion steps each, every run
changing one thing, and played each result against the starting state.

| Run | 1v1 goal share vs the starting state (384 games) |
|---|---|
| unchanged (twice) | 0.530, 0.536 |
| **all 27 helper terms off: goals only** | **0.653** |
| only the small helper terms off | 0.471 |
| no self-distillation teacher (see 5) | 0.434 |
| learning rate doubled | 0.523 |
| eight other changes to rewards, league and practice starts | 0.499 to 0.560 |

Played directly against the two unchanged runs, the goals-only run took 0.588 and 0.616 in 1v1 (512 games each) and
was level in 2v2 and 3v3 (0.47 to 0.54, 256 games a mode). It has kept improving since.

What this does not show: that shaping is useless. This network learned its basics *on* the shaped reward. The
measurement is that at this stage the helper terms were holding 1v1 back, and that removing only some of them was
worse than removing all.

## 2. A practice start only helps if its last step already pays

Under a goals-only reward, a drill teaches something only if the situation it creates leads to goals often enough
for the value to travel backwards.

- Our first flip-reset start put a car in the air below the ball. Policies got the reset in about a third of
  starts, and a shot followed in 0.1 to 0.4 percent of them, before and after 1 billion steps of training with it.
  The start gave the reset but the finish was out of reach, so nothing was learned.
- A chain with every stage as its own start (take-off, approach, wheels about to meet the ball, and the finish with
  a dodge in hand) pays from the first day: the same policy scores from the finish stage in about 75 percent of
  starts and from the reset stage in about 30 percent.
- A simulator detail that silently breaks such starts: a car that a state setter places in the air has never jumped,
  so it keeps its dodge for ever. A realistic start has to mark the car as having jumped and set its air time.
- What we have **not** achieved: after 1.6 billion steps with these starts, flip resets in real games are still
  about 0.003 per car-minute, against 0.34 for an opponent that uses them constantly. The policy finishes the play
  when it is handed the situation and does not yet create it. We do not know whether more time, a larger share of
  starts, or something else is needed.

These numbers were measured with the drills and counters as first written for our own trainer; this package is
their rewrite for RLGym (see [drills.md](drills.md) and [skills.md](skills.md)).

## 3. A few ticks of reaction time are worth a whole generation

The same policy against itself, with one side's actions landing 7 physics ticks late (the decision step is 8
ticks): **0.81 to 0.19** in goal share over 256 one-versus-one games. For scale, that is about the margin by which
our current network beats the one that taught it.

Two consequences we learned the hard way:

- If you train against a copy of somebody else's bot, give the copy the timing the real bot has. A copy that reacts
  faster or slower than the original is a different opponent, and results against it can be badly wrong in either
  direction.
- Small unfair edges make good sparring partners. Our league includes the policy's own average playing its most
  likely action while the learner's actions land one or two ticks late.

## 4. One pass over the batch or two: about a wash

Our update makes two passes over a batch of 512,000 samples in minibatches of 16,384, which is 63 optimiser steps
on data from one policy. A fair worry is that the last steps are far off-policy. We tested one pass in minibatches
of 49,152 (11 steps), with the learning rates raised until the policy moved as far per update as before, since at
the same rate fewer steps simply means a smaller update.

At equal training steps the one-pass run scored 0.478 in 1v1 (512 games), 0.449 in 2v2 and 0.506 in 3v3 (256 a
mode) against the two-pass run. It trained 1.6 times as many steps per hour. So: slightly worse per step, slightly
better per hour, and not worth changing either way for us.

## 5. A teacher made from the policy's own past helps

During training we add a small penalty (weight 0.05) on the distance between the policy and a fixed earlier
version of itself, the average of its weights over 750 updates. Removing it was the clearest loss in the table
above (0.434). Using the policy's own average as that teacher, in place of the previous generation's network, won
0.598 in 1v1 over 512 games at equal steps.

## 6. Watch the size of the critic's weights

With layer normalisation after each hidden layer, the scale of a layer's weights does not change what it computes,
but Adam moves every weight by about the learning rate per step whatever its size. So as the weights grow, the
layer changes more slowly relative to itself.

In our run the policy's weight matrices settled at 1.7 times their starting size within a billion steps. The
critic's, trained at ten times the learning rate, were at 7.0 times after 10.5 billion steps and still growing in
a straight line. We have not yet measured whether this costs strength; a run with a faster critic is in progress.

## 7. Measuring is harder than it looks

- Two deterministic policies in a deterministic simulator replay the same game. Head-to-head games need a source
  of variation; we let each car keep its previous decision with a small probability.
- A policy that samples its actions is far weaker than the same policy playing its most likely action. Opponents in
  a league should include the second kind, because that is what a deployed bot does.
- Per-game win rate exaggerates small differences. Goal share over a few hundred games has been the steadier
  measure for us.
- Results against a simulator copy of another bot and against the real bot can disagree. When it matters, play the
  real one.

## 8. A larger network, taught by the smaller one

Our fifth network had 2.8 million policy weights and plain ReLU layers. After 100 billion steps a fifth of its
hidden units never fired. Its successor has 20.8 million policy weights and layer normalisation, and started by
imitating the old one before training on its own. After 11 billion of its own steps it takes 0.82 of the goals
from its teacher in 1v1.
