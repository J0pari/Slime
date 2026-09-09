                                       Identity Increases Stability of Neural Cellular Automata

                                                                         James Stovold

                                              Lancaster University Leipzig, Nikolaistraße 10, 04109 Leipzig, Germany
                                                                    j.stovold@lancaster.ac.uk

arXiv:2508.06389v2 [cs.NE] 3 Nov 2025                                Abstract                                drift and inheritable mutations; Kvalsund et al. (2025)
                                                                                                             used the NCA model to develop active sensing in artificial
                                          Neural Cellular Automata (NCAs) offer a way to study               organisms; and Chow and Bentley (2025) use an NCA-
                                          the growth of two-dimensional artificial organisms from a          grown artificial organism to study how evolution conserves
                                          single seed cell. From the outset, NCA-grown organisms             part of the genetic code.
                                          have had issues with stability, their natural boundary often
                                          breaking down and exhibiting tumour-like growth or failing            From the outset, however, there were issues with the
                                          to maintain the expected shape. In this paper, we present a        stability of NCA-grown organisms, with Mordvintsev et al.
                                          method for improving the stability of NCA-grown organisms          (2020) proposing different training approaches to improve
                                          by introducing an `identity' layer with simple constraints         the stability (including damaging the growing organisms
                                          during training.                                                   during training). There were many problems associated with
                                                                                                             the rotation of NCA-grown organisms, with Mordvintsev
                                          Results show that NCAs grown in close proximity are more           et al. (2022) and Randazzo et al. (2023) focussed on
                                          stable compared with the original NCA model. Moreover,             developing new training techniques to address this. In this
                                          only a single identity value is required to achieve this increase  paper we consider the problem of instability stemming from
                                          in stability. We observe emergent movement from the                multiple organisms growing in close proximity.
                                          stable organisms, with increasing prevalence for models with
                                          multiple identity values.                                             When multiple NCA-grown organisms are close together,
                                                                                                             there is a tendency for the organism's `natural' boundary to
                                          This work lays the foundation for further study of the             break down (see fig. 1), with tumour-like growths forming
                                          interaction between NCA-grown organisms, paving the way            and trying to grow into new organisms (as if seeded
                                          for studying social interaction at a cellular level in artificial  by the user). This is problematic for any study into
                                          organisms.                                                         social interactions between NCA-grown organisms. This
                                                                                                             is not dissimilar to the problem of cellularity discussed
                                       Submission type: Full Paper                                           at length by Ray when building the Tierra system (Ray,
                                       Code/Videos available at: https://github.com/                         1992, 1993), where the introduction of a cell boundary
                                       jstovold/ALIFE2025                                                    decreased the brittle nature of artificial organisms in the
                                                                                                             Tierra system. In this paper, however, we are considering a
                                                             Introduction                                    multi-cellular organism, where the boundary is composed of
                                                                                                             many different cells. The problem of identity in a distributed
                                       While neural networks had been used to implement the                  system has been pondered for many years, including the
                                       rule of a cellular automata before (Li and Yeh, 2001),                classic Ship of Theseus problem1 (Chisholm, 2004), and
                                       the inclusion of modern neural network pipelines made                 also in Hofstadter's Ant Fugue (Hofstadter, 1979).
                                       Mordvintsev et al.'s (2020) Neural Cellular Automata
                                       (NCA) more viable as a research tool for studying                        We extend the original NCA model by introducing
                                       artificially-grown organisms.                                         an extra channel to the state representation, dubbed the
                                                                                                             `identity' layer (adapting the `environment' layer approach
                                          The NCA model offered the prospect of being able to                used in our previous work (Stovold, 2023)), and train
                                       study artificial organisms grown at the cellular level with           the NCA to produce an organism with its own identity
                                       comparative ease, subjecting the organisms to different               (represented by the value in the identity layer of each cell).
                                       manipulations or environmental changes. For example,                  By giving the NCAs different identities, we hoped they
                                       Cavuoti et al. (2022) showed that an organism could be                would be able to better distinguish between themselves and
                                       `taken over' by adversarial cells, effectively mimicking the
                                       introduction of viruses into a host; Stovold (2023) showed                1or "Trigger's Broom" problem, if you're a fan of British TV
                                       that an organism could grow into different shapes based
                                       on genetic information coded into the seed cell, and that
                                       grown organisms could respond to environmental signals;
                                       Sinapayen (2023) demonstrated that NCAs exhibit genetic
                                                                Parameter                   Values
                                                                Seed Time                   [0, 10, 50, 100, 150, 200, 250]
                                                                Lateral Distance            [6, 9, 12, 15, 18]
                                                                Vertical Offset (per seed)  [0, 5, 10, 15]

                                                                Table 1: Parameter values varied during the experiments.

Figure 1: Breakdown of NCA-grown organism's natural                For the work in this paper, we extend the original model
boundary, with tumour-like growths sprouting from the           by adding an extra channel (i.e. 17 channels instead of 16)
original organism.                                              where the extra channel serves as our `identity' layer. All
                                                                models are trained using the same approach as the original
others and, as such, reduce the likelihood of the organism      NCA paper (Mordvintsev et al., 2020), albeit with a larger
breaking down.                                                  pool size (12 instead of 8).

   Our results show that the inclusion of this identity            We train three models to produce a gecko image of
increases the stability of the organisms compared with the      the same size and shape. The models have the following
original NCA model, but did not require multiple identity       constraints:
values to achieve this. The increased stability permits the     Model A acts as our control, and has no change from the
study of multiple organisms in close proximity for the first
time, resulting in the observation of emergent movement                      original NCA model except for the extra channel
from the organisms as they adjust their position to avoid                    and increased pool size.
each other. Finally, we observe that the inclusion of
multiple identity values increases the prevalence of observed   Model B is trained to produce the same value (1.0) on the
movement.                                                                    identity layer for every living cell.

                        Methods                                 Model C is trained to reproduce the value provided in the
                                                                             seed cell on the identity layer for every living cell
The work in this paper relies heavily on the Neural Cellular                 (in training we used three values for this: 0.0, 0.5,
Automata model. What follows here is a brief description                     and 1.0).
of the model but for full details the reader is directed
to (Mordvintsev et al., 2020).                                     In contrast to the read-only environment layer used by
                                                                Stovold (2023), all three models here are able to both
   A cellular automata can be defined as a grid of cells, each  read from and write to the identity layer in this work.
with an automaton that senses the state of the neighbours       During training, all organisms are alone in the environment,
and updates its state accordingly (Izhikevich et al., 2015).    meaning the only time they encounter other organisms is
Time proceeds in a discrete manner, with each cell updating     during testing.
simultaneously.
                                                                   For each of the models, we evaluate their ability to grow
   The Neural Cellular Automata (NCA) (Mordvintsev              stable organisms by seeding two organisms close together,
et al., 2020) extends a two-dimensional cellular automata       growing them for 1000 timesteps, then measuring how far
in the following ways: the (previously binary) cell state is    they deviate from an idealised final state.
replaced by a vector of reals, and the automaton (which
was previously implemented using a rule-based system or            We test the effect of varying three parameters: the
similar) is replaced by a neural network. In the NCA, the       elapsed time between the first and second seeds, the lateral
neighbourhood can be sensed using a convolutional layer         distance between the two seeds (i.e. horizontal difference
within the network. We train the neural network to update       in position), and the offset of the two seeds (i.e. vertical
the state of each cell based on its neighbouring cells, such    difference in position). The parameters are varied as per
that a particular macroscopic form will grow from a single      table 1, giving 560 permutations per model, or 1680 in total.
seed cell. See fig. 2 for a diagram showing one pass of the
NCA update step.                                                   To determine the idealised final state, we produced an
                                                                image which depicts the expected outcome should the
                                                                two organisms grow without problems. We compare this
                                                                idealised image to the RGBA layers of the grown organisms
                                                                using a standard error function (RMSE), and by calculating
                                                                and comparing the bounding box around all living cells
                                                                in the image. This gives us a broad indicator of when
                                                                breakdown occurs (as the error will be substantially higher),
                                                                as well as indicating any difference in position or size. It
                                                                is worth highlighting that the absolute values of the error
                                                                function and change in bounding box are not particularly
Figure 2: Diagram depicting one pass of our extended NCA update step (with 17 channels instead of 16). The diagram also
shows the structure of the neural network. Image adapted from (Mordvintsev et al., 2020), licenced under CC BY 4.0.

Figure 3: Example idealised comparison images. Left                       0.0075
image has distance of 6 and relative offset -15, right
image has distance 12 and relative offset 5. Note the        Error Range  0.0050
saturation occurring when the two images intersect, which
is a consequence of the production process, so unlikely to                0.0025                                      Model
occur in the grown organisms.                                             0.0000
                                                                                                                             A
meaningful here; the idealised image has some saturation                                                                     B
where the two organisms overlap, and it does not take                                                                        C
into account cells that are in a growing state, but not yet
mature (Mordvintsev et al., 2020), so there will always be                        18  15  12  9                       6
some discrepancy. As we are only using these values in an
indicative manner, this should not have any impact on the                             Lateral distance between seeds
results.
                                                             Figure 4: Graph showing the change in error range as a
                         Results                             function of lateral distance between seeds. The error range
                                                             is the difference between maximum and minimum errors in
As described in the methods section, we grew 1680 pairs      the distribution. The graph shows much larger ranges for
of organisms with different parameter values. After 1000     model A compared with models B/C.
timesteps we calculated the RMSE and change in bounding
box between the state of the NCA and an idealised image.

   For the organisms grown with model C, we specify the
same identity value in the seed as for model B (1.0) to
ensure parity between the behaviour of the two models. We
                Model: A                                            Model: B                                                        Model: C

0.0100

0.0075

Error (RMSE)0.0050
                                                                                        Error (RMSE)
0.0025

0.0000

        18  15  12        9  6  18                              15  12                                9       6             18  15  12        9  6

                                Lateral distance between seeds

Figure 5: Boxplots showing the distribution of error for each lateral distance, organised by model. Model A clearly demonstrates
larger variance and less consistent behaviour compared with models B/C.

consider the effect of changing the seed value separately at                                                  Distance: 12      Distance: 9   Distance: 6
the end of the results section.
                                                                                                      0.0100
Identity Layer Increases Stability
                                                                                                      0.0075
Both model B and model C exhibited increased stability
when growing two organisms in close proximity, compared                                               0.0050
with model A. Fig. 4 shows how the range of errors varies
as we bring the organisms closer together. An increase in                                             0.0025
error range results from a broader spread of error values,
implying less-consistent behaviour. It is clear from fig. 4                                           0.0000                    ABC           ABC
that model A has much larger range of error values at all                                                           ABC
distances except 18 (when the organisms were far enough                                                                            Model
part as to not interact).
                                                                    Figure 6: Boxplots showing the distribution of error for
   Perhaps unsurprisingly, as we move the seed cells closer         each model, organised by lateral distance (larger distances
together and the interaction between the grown organisms            omitted). We see significant differences (at 95% confidence,
increases, the behaviour of the system changes and we get           p < 0.01667 after Bonferroni correction) in all three
increased error values. Fig. 5 shows the effect of varying          distances, with small (> 0.526) to medium (> 0.67) effects
the lateral distance between the two seed cells. When the           (measured by Vargha-Delaney A measure). See table 2 for
seeds are far enough apart (distance 18) the grown organisms        details.
are far enough apart they don't typically interact. The small
amount of variance in these cases stems from the vertical           of consistency between the three models; from the plots in
offset between the two seeds--for each lateral distance,            fig. 6, we can compare the behaviour of the three models
there are vertical offset values where the two organisms are        for each lateral distance. Using the Mann­Whitney rank
closer together and could interact.                                 sum test, we see a significant difference2 in the distributions
                                                                    between models A and B for distances 9 and 12, and a
   Fig. 5 shows a clear difference in behaviour between the         significant difference between models A and C for distances
three models. Model A appears to be much less stable, with          6, 9, and 12. Interestingly, there is also a significant
less consistent behaviour at larger lateral distances, whereas
models B and C are much more consistent, showing a clear                2at the 95% confidence level, using the Bonferroni correction
pattern of behaviour that only varies subtly between the two             to give a p-value threshold of 0.01667
models.

   The boxplots in fig. 6 are another view of the same
data, but focussed on the closer distances and organised to
highlight the difference between models for each distance.

   From the plots in fig. 5 we can see the different levels
Distance  A/B              A/C          B/C                      1.0 on the identity layer of each living cell, and model C
12                                                               trained to reproduce whichever value is given on the identity
9         p = 1.91e-5      p = 0.0001   Not significant          layer of the seed cell to the identity layer of each living
6         A = 0.67         A = 0.65     Negligible               cell (trained using 0.0, 0.5, 1.0). For all the previous results,
          p = 5.34e-7      p = 0.0099   p = 0.002                model C organisms have been grown using a seeded identity
          A = 0.69         A = 0.60     A = 0.62                 value of 1.0 in order to match model B.
          Not significant  p = 3.15e-5  Not significant
          Negligible       A = 0.66     Negligible                  The increased error described in the previous section
                                                                 results from organisms in model C moving away from
Table 2: Results of Mann­Whitney rank sum test (p) and           each other. This results in a higher error rate as the
Vargha­Delaney effect magnitude test (A). Significance           grown organisms are not in the expected position in the
determined at 95% confidence, p < 0.01667 after Bonferroni       environment when compared to the idealised comparison
correction. Effect magnitude can be characterised as Small       image. To determine how much the organisms have moved,
(A > 0.526) or medium (A > 0.67), as per (Hess and               we calculate the bounding box of all living cells after 1000
Kromrey, 2004). Distributions shown in fig. 6.                   steps, and compare to the bounding box obtained from the
                                                                 comparison image.
difference between models B and C for distance 9.
   The lateral distance of 9 has the most interesting               Fig. 8 shows the change in position of the top-left of the
                                                                 bounding box per lateral distance for each model. From
behaviour, where all three models behave in significantly        this figure, the difference in behaviour between model A
different ways. Fig. 7 shows the error when varying the          and models B/C is increasingly clear, with the breakdown
offset values (i.e. the relative vertical position of the seed   in model A organisms causing large changes in position
cells) for lateral distance 9. The two seed cells were each      whenever the organisms interact. The difference between
offset between 0 and 15 pixels for each lateral distance. As     models B and C are less pronounced, with most of the
the seeds are introduced to an otherwise-empty environment,      model B organisms consistently staying still. The handful
only the relative vertical distance between the two seeds is     of `outliers' for model B are cases where the organisms did
important; as such, for fig. 7 we calculate the relative offset  move, which predominantly happened when the organisms
distance and plot the error accordingly.                         were grown closest together.

   Due to the asymmetrical nature of the target image (the          To get a better sense of what is happening, we calculated
gecko emoji, shown in fig. 3), there will be offset values       the ratio of change in area for the bounding box. These
where the two organisms have more prominent interaction          results reflect those presented in fig. 8, where model B
than others. This is reflected in the asymmetry of the           shows a few cases of changed area, but model C has more
graphs in fig. 7. We can see the particular offsets where        prominent changes, especially for distances 9 and 6. Due
the behaviour of models B and C diverge: relative offset         to the similarity between these two sets of results, we have
-10, -5, 0, to varying degrees.                                  omitted this figure due to space constraints.

   While model C exhibits a higher error rate than model            While this explains the increased error, what is more
B--which might suggest a less-stable model--in this case         interesting is that the organism is able to move at all.
we are only using the error as a proxy for whether the system    The training process does not include any movement, and
is stable. As mentioned in the methods section above,            doesn't include other organisms--the organism is trained
the comparison image is not perfect: when two organisms          in an empty environment. As seen in fig. 8 above, this
are grown close together, the cell values are unlikely to        emergent behaviour is more prominent in model C than in
sum in the same way the comparison image does. The               model B; there are many parameter values which result in
particular behaviour we are seeing in the data warrants          the organism moving in model C but not in model B.
further investigation, especially as the lateral distance of 9
is the particular distance where the two organisms are just         Fig. 9 shows two examples of the type of movement
close enough to be growing into each other's space, but not      observed in model C. In these particular examples, the
so close that they are unable to fully form.                     organisms move in model C but not in model B, allowing us
                                                                 to plot the error rate over the lifetime of the organisms. The
Model B/C Organisms Exhibit Emergent                             first organism (on the left of both (a) and (b)) grows in the
Movement                                                         expected place in the environment. The second organism (to
                                                                 the right) grows alongside. After the two organisms interact
In the scenario where two organisms were grown sufficiently      for approx. 350 timesteps (for (a)) or 175 timesteps (for (b)),
close to each other, a higher error rate was observed for        the second organism moves to avoid being in the same space
model C than for model B when compared to the idealised          in the environment as the first organism.
comparison image (fig. 3). The difference between these
models is minor, with model B trained to produce the value          Given that this behaviour is more prominent in model C
                                                                 (which was trained to reproduce multiple identity values)
                                                                 compared with model B (which was trained to produce a
                                       Model: A                  Model: B                   Model: C

                 0.006

Error (RMSE)     0.004

                 0.002

                 0.000                                  -15 -10 -5 0 5 10 15        -15 -10 -5 0 5 10 15
                             -15 -10 -5 0 5 10 15
                                                                   Relative offset

Figure 7: Boxplots showing the distribution of error for lateral distance 9 as we vary the relative vertical offset, split by model.
The differences between the three models is most pronounced between -10 and 5 for this lateral distance. The variance in
these distributions comes from the different seed times, which has only a minor impact on the behaviour of the models.

                             Model: A  Model: B    Model: C      detailing the behaviour of the organisms. In two tests,
                 20                                              the organism seeded with value 0.0 breaks up (as seen in
                                                                 many cases when the seed value is 0.0--see below). For
Position change  15                                              every other combination, however, one of the two grown
                                                                 organisms moves away from the other organism.

                 10                                                        Seed 1   Seed 2  Moves?
                                                                             0.0      0.0   Yes
                 5                                                           0.0      0.5   Breaks up
                                                                             0.0      1.0   Breaks up
                 0                                                           0.5      0.0   Yes
                                                                             0.5      0.5   Yes
                        12 9 6         12 9 6    12 9 6                      0.5      1.0   Yes
                                                                             1.0      0.0   Yes
                        Lateral distance between seeds                       1.0      0.5   Yes
                                                                             1.0      1.0   Yes
Figure 8: Boxplots showing the distribution of bounding
box top-left position change as we vary lateral distance, split  Table 3: Table showing all permutations of seed identity
by model. Larger lateral distances (where there is minimal       values for two seeds, along with a description of whether
interaction) have been omitted for clarity.                      the seeds result in model C organisms moving (based on
                                                                 manual observation). Parameters used are: lateral distance
                                                                 6, relative offset 5.

single identity value), a reasonable question to ask would be       To test whether this is consistent behaviour, we grew
whether the particular value of the identity has an effect on    another 3360 pairs of organisms: 1680 had the seed identity
the behaviour of the organisms grown using model C.              value set to 0.0, 1680 had it set to 0.5, and we have the
                                                                 original set of 1680 with it set to 1.0. As expected, there
   The parameter values with the largest difference in           was no difference in behaviour between the different seed
behaviour between model B and model C are: lateral               values for model A or model B.
distance 6, relative offset 5. We grew nine organisms with
these parameter values while varying the seed cell identity         Fig. 10 shows the difference in behaviour for model C as
value, to understand whether the value of the seeded identity    we vary the seed identity value for the first-grown organism.
has an effect in the interaction between grown organisms.        From these plots, we can see that seed value 0.0 tends to
                                                                 exhibit higher error rates compared with other seed values.
   Table 3 shows the outcome of this test, where each
permutation of seed value is listed alongside a comment
              0.0100                                            Model                   0.0100                                  Model
              0.0075                                                                    0.0075
              0.0050                                                   B                                                               B
Error (RMSE)                                                           C  Error (RMSE)                                                 C

                                                                                        0.0050

              0.0025                                                                    0.0025

                      0  250                 500   750          1000                            0  250          500        750          1000

                                        Timesteps                                                               Timesteps

                                        (a)                                                             (b)

Figure 9: Top: two model C organisms growing into the same space causes one to move in a northwesterly direction (a) or
northeasterly direction (b) to better fit together. Snapshots at 100 and 900 timesteps. Bottom: error over the lifetime of the
organisms for model C compared with model B (where no movement was observed in these cases).

                         Distance: 18             Distance: 15            Distance: 12             Distance: 9             Distance: 6

              0.006

Error (RMSE)  0.004

              0.002

              0.000                          0.0 0.5 1.0        0.0 0.5 1.0                        0.0 0.5 1.0       0.0 0.5 1.0
                           0.0 0.5 1.0
                                                                  Seed Value

Figure 10: Boxplots showing the impact on model C organisms of varying the seed identity value (between 0.0, 0.5, and 1.0),
split by lateral distance. Higher error rates are observed for seed value 0.0 compared with other values.

This is in line with our observation above about organisms                organisms tend to break up (leaving scattered detritus of
seeded with 0.0 breaking up more often. Due to the way                    living cells behind), the area change measure is often not
                    Seed Value: 0.0  Seed Value: 0.5  Seed Value: 1.0  consistently they have grown.
                                                                          The emergence of movement from the NCA organisms
                 9
                                                                       to avoid other organisms is a particularly interesting
Position change  6                                                     development, suggesting that identity and individuality are
                                                                       intrinsically linked even in artificial organisms. The only
                 3                                                     other work we are aware of which looks at movement
                                                                       of NCA organisms is (Kuriyama et al., 2022), where the
                 0                                                     authors introduce a gradient into the environment that the
                                                                       organism is able to follow. In the present work, however, the
                    12 9 6           12 9 6           12 9 6           behaviour is untrained and emerges only when the organism
                                                                       is close to another organism. The logical next step for
                    Lateral distances between seeds                    this line of thinking would be to analyse the behaviour of
                                                                       the organisms using Krakauer et al.'s (2020) information-
Figure 11: Boxplots showing the bounding box position                  theoretic view of individuality. The main challenge with
change for three lateral distances, split by seed identity             this will be defining the environment when we have two
value. Seed value 0.0 shows much larger variance compared              individuals interacting in the same area. This is the primary
with other seed values.                                                area of future work.

particularly useful. Fig. 11 shows the change in position of              Once we have reliable mechanisms for growing artificial
the bounding box indicating that for seed 0.0, the organisms           organisms in NCAs, the next route we are looking into is
are more often further away from their starting point than             the use of adaptive identities (Stovold et al., 2014) and the
other seed values.                                                     adaptation of identity and individuality into a form of self­
                                                                       not-self distinction with potential routes to emergent self-
   This behaviour is most likely a consequence of either               awareness in NCA organisms (Mitchell, 2005). This route
training too many attractors too close together, or due to             allows us to answer fundamental questions about the role of
training the neural network to produce 0s on the output,               cellular interaction in macroscopic awareness.
rather than some underlying emergent behaviour from the
NCA model. It is, however, the subject of future work                     To conclude, we have shown that the introduction of a
looking at how the number of identity values increases the             simple constraint during the training of NCAs (in this case,
likelihood of certain behaviours emerging.                             producing a specific `identity' value on one of the NCA
                                                                       channels) is sufficient to increase the stability of grown
                       Discussion                                      organisms. By seeding two organisms in increasingly-
                                                                       close proximity, we saw that models with this constraint
Increasing the stability of NCA-grown organisms is an                  (models B and C) were closer to the expected macroscopic
essential step towards studying organism­organism inter-               shape compared with Mordvintsev et al.'s (2020) original
action, paving the way to cellular-level studies of social             NCA model (model A). Furthermore, when studying the
interaction between artificial organisms. What is required,            behaviour of model B and model C (which only differ in how
however, is a better way of characterising the stability of            many identity values the model was trained on), we observed
the organisms. It quickly became evident when studying                 emergent movement from the grown organisms when in
the output of the models that RMSE against an idealised                close proximity with another organism. This behaviour not
image only works if the organisms grow and stay in the                 only suggests an increase in individuality, but opens the door
expected location. We would argue that the organisms that              to studying related phenomena like awareness and adaptive
move to a new location in order to maintain their shape                identities.
are just as stable as those that don't, but the movement
causes an increase in error. The bounding box approach is a                                   References
step in the right direction, but still has issues--in particular
it considers all living cells rather than whole organisms,             Cavuoti, L., Sacco, F., Randazzo, E., and Levin, M. (2022).
meaning detritus left behind by a collapsed organism will                    Adversarial Takeover of Neural Cellular Automata. In Proc.
skew the results. One route might be to run a convolution                    ALIFE 2022: The 2022 Conference on Artificial Life.
of the single target image (i.e. one gecko) across the entire
image, but this would be difficult to reduce to a single               Chisholm, R. (2004). Person and object: a metaphysical study,
value that tells us how many organisms are alive and how                     volume 5 of Metaphysics : in 17 volumes.

                                                                       Chow, P. C. K. and Bentley, P. J. (2025). Development
                                                                             necessitates evolutionarily conserved factors. Scientific
                                                                             Reports, 15(1):9910.

                                                                       Hess, M. R. and Kromrey, J. D. (2004). Robust confidence
                                                                             intervals for effect sizes: A comparative study of cohen'sd
                                                                             and cliff's delta under non-normality and heterogeneous
      variances. In annual meeting of the American Educational
      Research Association, volume 1.

Hofstadter, D. R. (1979). Go¨del, Escher, Bach: an eternal golden
      braid. Random House, New York.

Izhikevich, E. M., Conway, J. H., and Seth, A. (2015). Game of
      Life. Scholarpedia, 10(6):1816.

Krakauer, D., Bertschinger, N., Olbrich, E., Flack, J. C., and Ay,
      N. (2020). The information theory of individuality. Theory
      in Biosciences, 139(2):209­223.

Kuriyama, S., Noguchi, W., Iizuka, H., Suzuki, K., and Yamamoto,
      M. (2022). Gradient Climbing Neural Cellular Automata. In
      Proc. ALIFE 2022: The 2022 Conference on Artificial Life.

Kvalsund, M.-K., Ellefsen, K. O., Glette, K., Pontes-Filho, S., and
      Lepperød, M. E. (2025). Sensor movement drives emergent
      attention and scalability in active neural cellular automata.
      bioRxiv, (2024.12.06.627209).

Li, X. and Yeh, A. G.-O. (2001). Calibration of cellular automata
      by using neural networks for the simulation of complex urban
      systems. Environment and Planning A, 33(8):1445­1462.

Mitchell, M. (2005). Self-awareness and control in decentralized
      systems. In AAAI spring symposium: Metacognition in
      computation, pages 80­85.

Mordvintsev, A., Randazzo, E., and Fouts, C. (2022). Growing
      isotropic neural cellular automata. In Proc. ALIFE 2022: The
      2022 Conference on Artificial Life, page 65.

Mordvintsev, A., Randazzo, E., Niklasson, E., and Levin, M.
      (2020). Growing Neural Cellular Automata. Distill.

Randazzo, E., Mordvintsev, A., and Fouts, C. (2023). Growing
      steerable neural cellular automata. In Proc. ALIFE 2023: The
      2023 Conference on Artificial Life.

Ray, T. S. (1992). Evolution, ecology and optimization of digital
      organisms. Technical report, Santa Fe Institute.

Ray, T. S. (1993). An evolutionary approach to synthetic biology:
      Zen and the art of creating life. Artificial Life, 1(1 2):179­
      209.

Sinapayen, L. (2023). Self-Replication, Spontaneous Mutations,
      and Exponential Genetic Drift in Neural Cellular Automata.
      In Proc. ALIFE 2023: The 2023 Conference on Artificial Life.
      MIT Press.

Stovold, J. (2023). Neural Cellular Automata Can Respond to
      Signals. In Proc. ALIFE 2023: The 2023 Conference on
      Artificial Life. MIT Press.

Stovold, J., O'Keefe, S., and Timmis, J. (2014). Preserving
      Swarm Identity Over Time. In Proc. ALIFE 2014: The 2014
      Conference on Artificial Life, pages 726­733.
