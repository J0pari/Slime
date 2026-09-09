Neural Cellular Automata: From Cells to Pixels

EHSAN PAJOUHESHGAR, EPFL, Switzerland
YITAO XU, EPFL, Switzerland
ALI ABBASI, EPFL, Switzerland
ALEXANDER MORDVINTSEV, Google Research, Switzerland
WENZEL JAKOB, EPFL, Switzerland
SABINE SÜSSTRUNK, EPFL, Switzerland

arXiv:2506.22899v2 [cs.CV] 26 Jan 2026

                                                                                                                          Scale 32 Growing - Jpeg Textures - old textures
Fig. 1. Summary of results. Our proposed method enables NCAs to generate high-quality outputs with minimal extra cost. Our method is applicable to
different NCA architectures and training targets. Left: Growing 2D shapes and images from a single seed; Middle: Texture synthesis in 2D; Right: Texture
synthesis on 3D Meshes. Online interactive demos are available at https://cells2pixels.github.io.

Neural Cellular Automata (NCAs) are bio-inspired dynamical systems in                 demands of real-time inference at high resolution. In this work, we over-
which identical cells iteratively apply a learned local update rule to self-          come this limitation by pairing an NCA that evolves on a coarse grid with a
organize into complex patterns, exhibiting regeneration, robustness, and              lightweight implicit decoder that maps cell states and local coordinates to ap-
spontaneous dynamics. Despite their success in texture synthesis and mor-             pearance attributes, enabling the same model to render outputs at arbitrary
phogenesis, NCAs remain largely confined to low-resolution outputs. This              resolution. Moreover, because both the decoder and NCA updates are local,
limitation stems from (1) training time and memory requirements that grow             inference remains highly parallelizable. To supervise high-resolution outputs
quadratically with grid size, (2) the strictly local propagation of information       efficiently, we introduce task-specific losses for morphogenesis (growth from
that impedes long-range cell communication, and (3) the heavy compute                 a seed) and texture synthesis with minimal additional memory and compu-
                                                                                      tation overhead. Our experiments across 2D/3D grids and mesh domains
Work done during internship at EPFL.                                                  demonstrate that our hybrid models produce high-resolution outputs in
                                                                                      real-time, and preserve the characteristic self-organizing behavior of NCAs.
Authors' Contact Information: Ehsan Pajouheshgar, ehsan.pajouheshgar@epfl.ch, EPFL,
Lausanne, Switzerland; Yitao Xu, yitao.xu@epfl.ch, EPFL, Lausanne, Switzerland; Ali
Abbasi, a80.abbasi@gmail.com, EPFL, Lausanne, Switzerland; Alexander Mordvintsev,
moralex@google.com, Google Research, Zurich, Switzerland; Wenzel Jakob, wenzel.
jakob@epfl.ch, EPFL, Lausanne, Switzerland; Sabine Süsstrunk, sabine.susstrunk@epfl.
ch, EPFL, Lausanne, Switzerland.
2 · Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, and Sabine Süsstrunk

1 Introduction                                                                             NCA State                 LPPN Output

Complex self-organizing systems consist of numerous simple compo-                                        s  Hs-Ws-C               o  Ho-Wo-K
nents that interact through simple1 local2 rules, producing coherent
macro behavior without centralized control. In nature, such systems                        Fig. 2. Sample Output of Our Hybrid Model. The NCA evolves on a
manifest across many scales: elementary particles bind into atoms                          coarse 128 × 128 lattice (left); Our Local Pattern Producing Network (LPPN)
and molecules, which then assemble into the diverse materials we                           acts as a neural field decoder that decouples the render resolution from the
observe; a single fertilized cell undergoes differentiation to form                        NCA grid size. We sample this field at 1024 × 1024 and, without retraining,
a fully developed organism during morphogenesis; and neurons                               at 8192 × 8192 in the magnified inset (right).
coordinate and synchronize their activations to support coherent
cognitive functions. In all cases, the global structure emerges not                        domains, we show that LPPN improves output quality at high res-
from top-down planning but from the collective effect of countless                         olution with minimal overhead, while retaining the characteristic
simple local interactions.                                                                 self-organizing properties of NCAs, along with their efficiency and
                                                                                           interactive controllability. We also provide an interactive web demo
   Neural Cellular Automata (NCAs) make this principle learnable:                          (cells2pixels.github.io) that runs our trained models fully on-device
a small neural update rule, shared across cells, can grow images                           in the browser.
and shapes from a seed [Mordvintsev et al. 2020; Sudhakaran et al.
2021] and synthesize textures [Niklasson et al. 2021; Pajouheshgar                         2 Related Works
et al. 2023]. Because NCAs learn an iterative self-organizing pro-
cess rather than a direct mapping, they naturally exhibit robustness                       2.1 Neural Cellular Automata
and regeneration [Mordvintsev et al. 2020], generalize across dis-
cretizations and domains [Pajouheshgar et al. 2024b,a], and display                        Early computational explorations of self-organization trace back to
emergent spontaneous motion [Xu et al. 2024a], while encoding                              Alan Turing's reaction-diffusion model of morphogenesis [Turing
this process with a lightweight, compute-efficient neural network.                         1952] and John von Neumann's formulation of cellular automata
                                                                                           [Neumann 1966]. Both lines of work revealed that simple hand-
   In practice, however, current NCAs are trained on grids, volumes,                       crafted local rules, applied over time, can yield intricate global
or meshes with at most 104  105 cells [Mordvintsev et al. 2020;                            patterns. Carefully hand-designed CA and reaction-diffusion rules
Pajouheshgar et al. 2024b; Wang et al. 2025], which translates to                          have reproduced 2D and 3D morphogenesis and texture phenom-
spatial resolutions of about 64 × 64 to 256 × 256 depending on the                         ena [Fleischer et al. 1995; Gobron and Chiba 1999; Turk 1991], yet
space dimensionality. Scaling further is difficult: memory and com-                        the need for laborious manual rule search has long limited further
pute grow quickly with the number of cells, while information still                        exploration. Renewed attention to cellular automata began with
propagates locally (one neighborhood hop per update), requiring                            Gilpin [2019], which showed that a classical CA update can be writ-
more steps for long-range coordination. As a result, it remains un-                        ten as a shallow convolutional neural network and trained with
clear how to obtain high-resolution, high-quality outputs without                          back-propagation. Extending this idea, Mordvintsev et al. [2020]
abandoning locality or efficiency.                                                         parameterize the update rule by an MLP, and use fixed convolutional
                                                                                           kernels for cell interaction modeling, giving rise to NCA.
   We address this limitation by decoupling dynamics from appear-
ance. We evolve the NCA on a coarse lattice, but render a continuous,                         NCAs have been applied to morphogenesis and growth model-
high-resolution output with a lightweight coordinate-based decoder,                        ing: Mordvintsev et al. [2020] grows 2D images from a seed while
LPPN, which acts as a neural field conditioned on the local NCA                            demonstrating regenerative capabilities, Sudhakaran et al. [2021]
state3. Concretely, LPPN maps a locally interpolated cell feature and                      extends the idea to 3D voxel grids by using 3D convolutions, and
an intra-primitive coordinate to appearance at arbitrary query loca-                       Sudhakaran et al. [2022] conditions the update rule to grow multiple
tions, decoupling the render resolution from the NCA lattice size as                       targets. NCAs have also been successful for texture synthesis: Niklas-
shown in Figure 2. We train the NCA and LPPN end-to-end; LPPN                              son et al. [2021] demonstrated exemplar-based texture synthesis
adds only 20  30% extra parameters. Since all recurrent updates                            with emergent motion, Pajouheshgar et al. [2023] added control-
happen at low resolution4, training remains memory-efficient and                           lable dynamics, Pajouheshgar et al. [2024a] improved stability and
fast, and inference remains real time.                                                     scale control via noise initialization, Pajouheshgar et al. [2024b]
                                                                                           generalized NCAs to meshes via non-parametric message passing,
   We evaluate our hybrid NCA+LPPN across four settings: mor-                              and Larsson et al. [2025]; Wang et al. [2025] extend this framework
phology growth from a seed, 2D PBR texture synthesis, texture                              to 3D to synthesize volumetric textures. Beyond morphogenesis
synthesis on meshes, and 3D volumetric texture synthesis. Across                           and texture synthesis, NCAs have also been used for generative
                                                                                           modeling [Kalkhof et al. 2024; Otte et al. 2021; Palm et al. 2022;
1By simple rules, we mean interaction mechanisms with low descriptive complexity           Tesfaldet et al. 2022] and discriminative tasks such as classification,
relative to the complexity of the resulting emergent behavior.
2While locality of interactions is not strictly required in a complex system, every self-
organizing system observed in nature relies on local interactions among its components.
3From the biology perspective, the NCA provides the high-level blueprint that deter-
mines where structure should emerge, while the LPPN sidesteps the intricate biochem-
ical refinement processes and instead renders those details in a single pass.
4This mirrors the common strategy in modern vision models (e.g., transformers or latent
diffusion) of performing the expensive computation on a compact set of tokens/latents
rather than at full pixel resolution.
                                                                                        Neural Cellular Automata: From Cells to Pixels · 3

          NCA Update           Primitive Rendering                                      Local Paern Producing Network (LPPN)
                                  (Rasterization)
... st-1     st  st+1 ...
                                     si
                                                                            u(p)  3                                      o(p)  K
          p
                                                                               Local                                               Albedo
                                                                           Coordinates

                               p                                           s¯(p)  C                                           Surface
                                                                                                                              Normal

                           sk                       s Locally Averagedj                                                  ...
                                                                           Cell State

      NCA Cell Laice

Fig. 3. Hybrid NCA + LPPN Overview. Left: The NCA operates on a coarse lattice of cells (in this example vertices of a mesh) Center: A sampling point p
(red dot) inside a triangle primitive, whose vertices correspond to NCA cells ,  ,  . The local coordinate  (p) expresses the point's position inside the
primitive, while the locally averaged cell state ¯ (p) is obtained by interpolating the surrounding cell states. Right: A shared lightweight MLP, LPPN, receives
(¯ (p),  (p) ) as input and outputs the appearance features, such as color and surface normal, at point p.

segmentation, and robust representation learning [Guichard et al.          2021; Pajouheshgar et al. 2024a, 2023; Sudhakaran et al. 2021], or a
2025; Kalkhof et al. 2023; Randazzo et al. 2020; Sandler et al. 2020;      convolution-like operator for meshes [Pajouheshgar et al. 2024b].
Xu et al. 2024b].                                                          The adaptation function A is commonly parameterized by a Multi

   Despite this progress, most NCA models still operate on relatively      Layered Perceptron (MLP), which sometimes includes a stochastic
modest cell counts, which limits the achievable output resolution          term [Mordvintsev et al. 2020; Niklasson et al. 2021; Pajouheshgar
and detail. Scaling further is impeded by GPU memory limits, slow
information propagation in large grids, and the locality-breaking          et al. 2024b, 2023; Xu et al. 2024b]. The state of cell ,  , is updated
global operations often introduced as work-arounds. We overcome            as in Equation. 1.
this limitation by keeping the NCA coarse while using a lightweight,
local neural field decoder to render fine detail at arbitrary resolution.               + =  + A (Z ( ,  )) · ,   N ().       (1)

2.2 Neural Fields                                                          N () denotes the set of neighbors of cell . After evolving for  steps,
                                                                           the cell states  of all cells can be extracted for further downstream
Neural fields (a.k.a. implicit neural representations) model signals as    applications. We forward the resulting cell states to the Local Pat-
continuous coordinate-based functions [Stanley 2007]. They have
become central in vision and graphics, e.g., DeepSDF for shape             tern Producing Network, which transforms the low-resolution NCA
fields [Park et al. 2019], NeRF for view-dependent radiance and            features in  into high-resolution detailed output.
density [Mildenhall et al. 2020], and SIREN for high-frequency signal
fitting via sinusoidal activations [Sitzmann et al. 2020]. While neural    3.2 Local Pattern Producing Network (LPPN)
fields are resolution-free, semantic structure is typically embedded
in dense weights, making them less interpretable and harder to             The LPPN converts the coarse NCA lattice into a high-resolution
manipulate.                                                                field by evaluating a small MLP decoder network at every sampling
                                                                           point p, as shown in Figure 3. For each p, the decoder is supplied
   Our hybrid design combines the strengths of both paradigms: the         with:
NCA produces an explicit, local, editable lattice of features, while
LPPN uses these features as conditioning to render a continuous                (1) locally interpolated cell state ¯ (p)  R , which aggregates
high-resolution field. This preserves locality and interactivity while              the states of the cells in p's surrounding primitive.
lifting the resolution bottleneck.
                                                                               (2) local coordinate vector  (p) that encodes the relative position
3 Method                                                                            of p inside its primitive.

3.1 Preliminaries - Neural Cellular Automata                                  Below, we define the primitive associated with sampling points5
                                                                           and specify how the locally interpolated state and local coordinates
Neural Cellular Automata (NCAs) models the temporal evolution of           are computed.
a set of cells on a grid. Let ,  R denote the -dimensional vector
storing the -th cell state at time . At each time step, cells perceive     3.2.1 Primitives. A primitive  consists of the cells surrounding
their neighbors in the Perception stage Z, to collect the informa-         a sampling point p along with the geometric shape determined by
tion of their local neighborhood. Based on this information, cells         their positions. A primitive's shape is determined by the underlying
adapt to the environment by updating the states in the Adaptation          lattice; on a 2D quad grid, for instance, the de facto primitive would
stage A. The perception Z is often instantiated using convolu-             be a rectangle. Figure 4 illustrates four common primitive types. For
tion for 2D and 3D grids [Mordvintsev et al. 2020; Niklasson et al.          , let  denote the position of the -th cell in the primitive. We
                                                                           call a set of weights  (p) that describe the position of any arbitrary

                                                                           5A sampling point may be the center of a rasterized pixel for 2D images or 3D meshes,
                                                                           or a 3D position for a NeRF-style volumetric rendering.
4 · Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, and Sabine Süsstrunk

              W        La ice: 2D Grid, ad Mesh                                              A=x+y+z                     La ice: Triangular Mesh

        xp             Primitive: Rectangles (4 points)                        x py                                      Primitive: Triangles (3 points)
                        - Coordinates: Bilinear Weights                                                                   - Coordinates: Barycentric
H                      u - Coordinates: Cartesian                                                                        u - Coordinates: Normalized Barycentric
                y
                                    xy                     x     y             z                                         (p)  =     x  ,  y  ,  z     ,  u(p) = 2(p) - 1
vi                                 HW                      W     H                                                                  A     A     A
                       i(p)     =          ,  u(p)  =  2(     ,     )  -  1                                                      (                 )

                                                                                     i(p)  =  tan        i  +  tan  i-1  La ice: Graphs, Voronoi
                                                                                                         2           2
                    W
    D                  La ice: 3D Voxel Grid                                                             vi - p          Primitive: Polygons (n points)

    p                  Primitive: Cubes (8 points)                                       vi-1                             - Coordinates: Mean Value
                        - Coordinates: Trilinear Weights                                                                 u - Coordinates: Normalized Mean Value
                       u - Coordinates: Cartesian                              p
H                                                                               i i-1                                                         i(p)
                                                                                                                                              j(p)
    xy                          xyz                    x yz                                                              i(p) = ui(p) =                  ,  u(p) = 2(p) - 1
       z
vi                     i(p)  =  HWD     ,  u(p)  =  2( W ,    ,  D) -     1                                 vi
                                                            H
                                                                               vi+1

                   Fig. 4. Representative examples of primitives. Vertices correspond to neighboring cells of an arbitrary sampling point p.

p inside the primitive with respect to the primitive's vertices a                    Cell Laice                          Raw u-Coordinates               Transformed u-Coordinates
-coordinate if it satisfies the following three conditions:
                                                                                                                                                                                  (a)
    (1) Partition of unity:    (p) = 1.
    (2) Non-negativity  (p)  0.                                                                                                                                                   (b)
    (3) Linear precision: p =    (p)  .
                                                                               Fig. 5. Local coordinate transformations. Raw -coordinates visualized
   While triangles admit a unique -coordinate (barycentric), higher-           as RGB suffer discontinuities at primitive boundaries. (Top Right) Applying
order primitives offer multiple valid choices. For the rectangular             trigonometric functions to Cartesian coordinates enforces 0 continuity at
primitive, a valid choice for -coordinates is the bilinear interpola-          boundaries for rectangle primitives. Bottom Right (a): Sorting the barycentric
tion weights. Figure 4 illustrates the most common variants adapted            coordinates enforces 0 continuity but yields an imbalanced dynamic range
in this paper for four different types of primitives. With the chosen          (red color dominates). (b) Applying a remapping equalizes the range, giving
-coordinate system in place, we obtain the first LPPN input by                 a uniform, continuous positional field that is easier for the LPPN to digest.
averaging the cell states within p's primitive :
                                                                               positional cues, we apply simple, primitive-specific transforms to
                                                                               the raw -coordinates. For rectangle and cube primitives we keep
                                                                               the Cartesian coordinates but encode them with a sinusoidal basis
                       ¯ (p) =  (p)                                       (2)  of the first  harmonics,

                                                                                      aug = sin(), cos(), . . . , sin(), cos() . (3)
                                                                               This encoding is continuous across primitive boundaries (Fig. 5,
3.2.2 Local Coordinates. Rather than feeding the full -coordinates             top-right) and is injective on the interior of each primitive.
dirtetctly to the LPPQ Nu , we transform them to a more compact local co-
ordinate  (p), which is easier for the decoder to digest. For rectangle           For triangular or polygonal primitives, the raw -coordinates are
and cube primitives we use axis-aligned Cartesian coordinates for              discontinuous at primitive boundaries because neighboring faces list
. This representation fully determines the point's location inside             their vertices in arbitrary orders. To eliminate this order-dependence
the primitive with fewer dimensions than the eight (or four) dimen-            we first sort the -coordinates (barycentric/mean-value) in descend-
sional -coordinates. For triangular meshes and general polygonal               ing order, making the largest weight always the first component.
primitives we retain the -coordinates. In every case we rescale                This consistent reordering makes the local coordinate field 0 across
each component so that  (p)  [-1, 1], whereas the original -                   primitive boundaries, as illustrated in Figure 5,(a). Sorting, however,
coordinates lie in [0, 1]. This zero-centered range removes a poten-
tial bias in the decoder's inputs and empirically improves learning.
Figure 4 summarizes the specific choices of -, and -coordinates for
the primitive types considered in this paper. Before passing  (p) to
the LPPN, we optionally apply simple transformations that improve
its continuity and uniformity, as detailed next.

3.2.3 Ensuring 0 Continuity and Uniformity. Raw -coordinates
are piecewise smooth yet generally discontinuous at primitive bound-
aries, as shown in the middle column of Figure 5. On a 2D grid (top),
the Cartesian coordinates reset at each primitive, and on a trian-
gular mesh (bottom) the barycentric coordinates change abruptly
across edges. To provide the LPPN with smooth, well-conditioned
                                                                         Neural Cellular Automata: From Cells to Pixels · 5

skews the dynamic range of the different components of  (visible         and weight-shared, making the gradients computed from random
as the predominance of red in Figure 5(a)). To better balance these      crops representative of the entire output.
inputs and make it easier for the LPPN to utilize all coordinates, we
apply a simple monotone remapping to each sorted component so               Pseudo targets for PBR map alignment. For PBR texture syn-
that its values are spread more uniformly over [-1, 1]; see Appendix     thesis, the LPPN outputs 9 values corresponding to three maps:
for details. The resulting coordinates have comparable amplitude         albedo, normal, and HRA (height, roughness, ambient occlusion).
across components and remain continuous across primitives (Fig-          Applying the texture loss independently per map can lead to cross-
ure 5(b)). Although sorting makes the coordinates non-injective,         map misalignment [Pajouheshgar et al. 2024b]. To explicitly couple
LPPN also conditions on ¯ (p), which provides the missing context        different texture maps, we add pseudo targets: at each iteration we
and makes the joint input expressive enough for high-quality de-         randomly select three distinct channels from the union of all tar-
coding.                                                                  get maps, stack them into a synthetic 3-channel target, and apply
                                                                         the same multi-scale OT loss to the corresponding three generated
3.2.4 MLP Decoder. The LPPN is a lightweight MLP  that takes             channels. This channel mixing provides an explicit alignment signal
the locally averaged state and local coordinates of each sampling        across maps (validated in our ablations).
point (¯ (p),  (p)), and returns the desired attributes at the sampling
point:                                                                      Auto-correlation regularizer. To encourage long-range geo-
                                                                         metric structure, we optionally add an FFT-based auto-correlation
 (p) =   ¯ (p),  (p)        (4)                                          loss on early VGG feature maps, following correlation-based tex-
                                                                         ture constraints [Gonthier et al. 2022; Sendik and Cohen-Or 2017].
                        R,                                               We compute a channel-aggregated auto-correlation map for target
                                                                         and generated features and penalize their L1 difference; in prac-
where  is the number of channels in the output, e.g.  = 3 for            tice, applying this term only at the coarsest scale of our multi-scale
RGB-only, or  = 9 when the base color is augmented with surface          loss is sufficient. We enable this term only for textures with strong
normals, height, ambient occlusion, and roughness maps. Because          geometric structure and long-range correlations.
LPPN is evaluated only during loss computation, all recurrent NCA
updates still run on the coarse lattice. Therefore adding the LPPN       4.2 Morphology Loss
incurs negligible training overhead.
                                                                         We train an NCA to grow a target morphology from a single seed,
4 Loss Functions                                                         following Mordvintsev et al. [2020]. The target is an RGBA image
                                                                         whose alpha channel defines the desired shape; We pad the target
We design task-specific objectives for our two main settings: texture    into a larger transparent canvas to give the morphology room to
synthesis and morphology growth from a seed.                             expand. In our hybrid model, the NCA evolves on a coarse lattice
                                                                         and produces a living mask (one channel) that gates updates, while
4.1 Texture Loss                                                         LPPN decodes the NCA state into a high-resolution RGBA image.
                                                                         Our morphology loss is the (unweighted) sum of:
We supervise texture synthesis with a VGG-based texture loss built
on the relaxed optimal transport (OT) style loss of Kolkin et al.           (i) RGBA reconstruction on LPPN output. We apply 1 + 2
[2019], as adopted in prior NCA texture work [Pajouheshgar et al.        between the predicted and target RGBA images after masking both
2024b, 2023]. Concretely, we extract features from a fixed pretrained    the generated output and the target image by their corresponding
VGG16 [Simonyan and Zisserman 2015] at a few layers and match            alpha channels.
the distributions of generated and target features using OT with
additional moment matching (mean and covariance) [Gatys et al.              (ii) Shape loss on the NCA living mask. Since only living cells
2015]. We extend this baseline with (i) patch-based multi-scale su-      (and their neighbors) can update their state, directly supervising the
pervision, (ii) pseudo targets for multi-map PBR texture alignment,      living mask is essential to teach the NCA to expand from the seed.
and (iii) an auto-correlation regularizer. All losses are computed on    We bilinearly upsample the living mask to the render resolution and
images rendered from the LPPN output.                                    apply 1 + 2 against the target alpha channel.

   Patch-based multi-scale supervision. Prior work applies VGG              (iii) LPIPS for sharp outputs. Because growth is stochastic
on an image pyramid to capture texture statistics across multiple        and local, small variations and slight misalignments are common;
scales [Gonthier et al. 2022; Snelgrove 2017], but running VGG           optimizing only 1 + 2 tends to average over plausible outcomes and
on full-resolution pyramids becomes increasingly expensive for           produce blurry outputs. To counteract this, we add an LPIPS loss
high-resolution outputs. We therefore evaluate the OT style loss at      [Zhang et al. 2018] to encourage the model to generate perceptually
three scales (full, half, and quarter target resolution) using random    similar texture and appearance to the target. Since LPIPS expects
patches from the rendered output: at each scale we take random           three channels, we compute it by randomly selecting three of the
crops with the number and size of the crops increasing with the          four RGBA channels per batch element.
scale, resize each crop to a fixed input size (e.g., 256 × 256), and
compute the OT style loss. On the target side, we do not crop so as      5 Experiments
not to lose any information; instead we build a target pyramid by
resizing the exemplar per scale and caching its VGG features. This       We evaluate our hybrid NCA+LPPN framework on four representa-
patch-based multi-scale loss avoids the quadratic blow-up of feeding     tive settings: (i) Growing a Morphology, (ii) Synthesizing PBR
full-resolution pyramids to VGG and the cost scales primarily with       Textures, (iii) Synthesizing Textures on Meshes, and (iv) Syn-
the number of crops. Moreover, it is particularly well-suited to our     thesizing 3D Volumetric Textures. Table 1 summarizes the archi-
setting as both the NCA update rule and the LPPN decoder are local       tecture and training hyperparameters for each experiment.
6 · Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, and Sabine Süsstrunk

Table 1. Experiment configurations. We report NCA/LPPN architecture     Target                           Ours  Baseline
and training hyperparameters. Input dim = locally averaged NCA state +
                                                                                                         (NCA + LPPN) (NCA + Linear)
local coordinate encoding.

      Setting         Growing   PBR-2D       Mesh     Vol-3D

NCA   Domain          2D grid    2D grid   Icosphere  3D grid
      #Cells            962       1282        40k       643
      Channels            32        16         16        16
      MLP Width          256       128        128       128
      #Params            41k       10k        12k       12k

LPPN  Input dim        32+4       16+2       16+3      16+3
      Output dim ( )      4          9          3         4
      MLP Width           64        32         32        32
      #Params            11k        3k         3k        3k

Output Resolution       7682     10242      10242      5122
Train Iterations         20k        3k         3k        4k
Batch Size                8          2          2         4
Rollout Steps         [32, 96]                        [16, 48]
                                [32, 128]  [32, 96]

   We follow standard NCA training practices (checkpoint pool,                     Fig. 6. Morphology growth: comparison to the baseline.
stochastic updates, random rollout lengths, overflow regularization,
and gradient normalization). In all experiments we use a 4-layer        degrades detail or introduces artifacts, and removing LPIPS or the
SIREN MLP [Sitzmann et al. 2020] as LPPN (three sin layers + linear     sin/cos encoding leads to blurry outputs or patch discontinuities.
output), adding  25% parameters relative to the NCA.
                                                                        5.2 PBR Texture Synthesis
   Our primary baseline replaces LPPN with a single linear readout
from the locally averaged cell state (no coordinate input); matching    To synthesize 2D PBR textures the LPPN outputs a 9-channel field
the baseline parameter budget by increasing NCA channels yields         (albedo, normal, HRA). We apply a physics-based renderer to the
similar results, so we report the simpler baseline with matched         synthesized maps only for visualizations and not during training.
NCA parameter count. We do not compare to an NCA operating              This setting compresses  9 × 10242 target values into a model with
directly at render resolution, as its training memory requirements      13k parameters. For this experiment we use raw local coordinates
exceed current GPU memory limits. Full training details for each        for LPPN (no sin/cos), with similar observed performance.
experiment are provided in the Appendix. All figures are best
viewed digitally after zooming.                                            Figure 14 compares against the baseline and includes ablations
                                                                        on multi-scale supervision and pseudo targets, as well as the ef-
5.1 Growing a Morphology                                                fect of the auto-correlation loss. Multi-scale supervision improves
                                                                        global coherence, and pseudo targets improve cross-map alignment.
We train an NCA to grow a target morphology from a single seed.         For textures with strong geometric regularity, the auto-correlation
Targets are provided at 512×512 and embedded in a 768×768 trans-        term improves long-range structure and shows that NCA dynamics
parent canvas to give the morphology room to expand. The NCA            can produce rigid, highly structured patterns under suitable long-
evolves on a 96×96 grid and exposes a living channel that gates         range supervision despite their spontaneous dynamics and fluid-like
updates; LPPN decodes the final state into a 768×768 RGBA image         nature [Pajouheshgar et al. 2023]. We refer to the Appendix for ad-
supervised by the morphology loss (Sec. 4.2). We encode local 2D        ditional results and visualization of all PBR maps.
coordinates with sin/cos pairs to enforce 0 continuity (Sec. 3.2).
To improve robustness to poor local minima, we periodically flush          Figure 9 shows the effect of swapping NCA and LPPN between
the pool and restart the learning-rate schedule.                        trained models, revealing a coarse-to-fine division of labor: the NCA

   Figure 6 compares our method to the primary baseline. Our out-
puts have consistently sharper shapes and structures with finer
appearance details, while the baseline tends to blur boundaries and
wash out high-frequency content. Figure 8 visualizes the evolution
over time: starting from a single seed, the learned dynamics expand
outward and naturally grow the target silhouette and morphology.

   Figure 7 highlights that the learned rule defines a stable self-
organizing process: after severe damage the system re-grows miss-
ing regions and returns to the same attractor; swapping model pa-
rameters at test time produces a gradual morph from one converged
morphology to another. Finally, Figure 13 confirms the role of key
components: removing LPPN or its coordinate input substantially
                                                                                Neural Cellular Automata: From Cells to Pixels · 7

determines the coarse geometric layout, while the LPPN contributes      T  T+8  T+16  T+32  T+64  T+512
fine-scale appearance. Figure 10 shows that rendering at resolutions
different from training remains stable, and quality generally im-       Fig. 7. Top: Regeneration after damage. Bottom: Morphing between targets
proves as the output resolution increases. This provides a practical    by switching model parameters.
compute­quality knob: higher-resolution evaluation costs more but
yields sharper and more detailed results.                               (NeRF-style ray-marching through the learned texture field). All
                                                                        models are trained with an 8× LPPN scale, but we default to smaller
5.3 Volumetric 3D Texture Synthesis                                     scales for real-time performance on edge devices (typically 4× for
We synthesize volumetric textures with an NCA on a 643 grid repre-      morphology/PBR and 2× for volumetric); in particular, volumetric
senting a cube in [-1, 1]3. To render a 2D image, we use NeRF-style     rendering at 8× may not run in real time without a strong GPU.
volumetric rendering [Mildenhall et al. 2020]: along each camera
ray we sample points uniformly in the cube-ray intersection region,     6 Conclusion
trilinearly interpolate the local NCA state, and query LPPN to obtain
RGB color and a scalar density, followed by a softplus nonlinear-       We introduce a hybrid self-organizing framework that pairs Neu-
ity to ensure non-negative values for volumetric integration. We        ral Cellular Automata with a lightweight Local Pattern Producing
then composite samples along the ray using standard emission­           Network to decouple NCA grid size from output resolution. The
absorption to produce the rendered image, which is supervised by        NCA guides pattern formation through local updates on a coarse
the texture loss in Sec. 4.1. We use an orthographic camera to reduce   lattice, while the shared LPPN transforms a locally interpolated
perspective distortion. We restrict viewpoints to respect the target's  cell state together with continuous local coordinates into appear-
top­down and left­right symmetries. Figure 11 shows that our            ance attributes at any requested scale. Our proposed task-specific
model produces more coherent structure and substantially sharper        loss functions designed for high-resolution outputs further improve
details than the baseline.                                              the quality without exhausting memory and compute budgets. We
                                                                        demonstrate that NCA, paired with our LPPN, is capable of gener-
5.4 Texture Synthesis on Meshes                                         ating high-resolution images and textures in real time. Our hybrid
                                                                        formulation enables NCA to scale to practical resolutions without
We adopt MeshNCA [Pajouheshgar et al. 2024b], where cells live on       losing its characteristic properties, such as robustness, controlla-
mesh vertices and perception is implemented via message passing         bility, and efficiency, paving the way for interactive, deployable
over mesh connectivity using a spherical harmonics basis. Following     self-organizing systems.
Pajouheshgar et al. [2024b], we train on an icosphere (40k vertices)
and apply the trained model to unseen meshes at test time without          Limitations. While LPPN substantially improves output resolu-
retraining. In our hybrid setting, LPPN decodes locally averaged        tion, it conditions only on intra-primitive coordinates and locally
vertex states together with continuous surface coordinates to an        interpolated state (no access to cells outside the enclosing primitive),
RGB texture at arbitrary sampling points.                               which can produce faint primitive-aligned patch artifacts.

   For stable decoding across triangle boundaries, we apply a simple       Future work. A natural next step is to extend our morphology
barycentric-coordinate preprocessing (sorting + remapping). Dur-        setup from 2D objects to growing full 3D assets while preserving
ing training only (on the icosphere), we accelerate loss evaluation     regeneration and controllability. On the optimization side, our tex-
by replacing perspective rasterization with cached Lambert (equal-      ture losses already operate on patches; integrating this more tightly
area) projections of random spherical patches. These projections        with LPPN evaluation (rendering only the queried regions used by
also reduce curvature and perspective distortions and allow us to       the loss) could further reduce memory and enable training at even
precompute and reuse the barycentric coordinates and face indices       higher effective resolutions (e.g., 4K and beyond). Beyond synthesis,
for a fixed patch size, making training faster than repeatedly in-      the compactness of NCA+LPPN suggests applications in learned
voking a full rasterizer. For the results shown in the paper and for    compression of images and textures. Finally, one could explore al-
testing on arbitrary meshes, we use a standard rasterizer with a        ternative formulations better suited to isotropic settings, such as
perspective camera. Figure 12 shows improved sharpness and de-          coordinate-free LPPN variants that avoid explicit intra-primitive co-
tail over the baseline and highlights the importance of our local       ordinates and instead condition on both the interpolated and nearest
coordinate transformation to prevent triangle-aligned artifacts.        neighbor cell states.

5.5 Online Demo

We provide an interactive WebGL demo at cells2pixels.github.io that
runs our trained models fully on the GPU inside the browser (im-
plemented with SwissGL). The core shader programs include: NCA
updates, LPPN decoding/rendering, and interactive edits (erasing
regions or inserting seeds). Additional controls let users adjust the
simulation speed and the LPPN sampling scale. The demo includes
three modes corresponding to our main experiments: morphology
growth (direct RGBA output), 2D PBR textures (9-channel maps
visualized with a simple PBR shader), and 3D volumetric textures
8 · Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, and Sabine Süsstrunk

T=0            T=8    T=16                                                                                     Ours            Baseline

                                                                           Target                        (NCA + LPPN) (NCA + Linear)

                            T=0         T=8      T=16

T=32           T=512

                            T=32

T=64

                            T=64                 T=512

Fig. 8. NCA growth from a single seed to the target morphology over time.

                            LPPN Model

NCA Model

                                                                           Fig. 11. 3D texture synthesis results, and comparison to the baseline.

                                                                           Target                        Ours        Baseline  No Coordinate

                                                                                                         (NCA + LPPN) (NCA + Linear) Transformation

Fig. 9. Swapping NCA and LPPN between trained models. Coarse layout
follows the NCA, while fine detail follows the LPPN.On the diagonal, we
show the original (unswapped) models.

           1x         2x    3x               4x

Fig. 10. Increasing the LPPN evaluation resolution improves details and    Fig. 12. Mesh texture synthesis results, comparison with the baseline, and
sharpness, enabling a compute­quality trade-off (training setting: 8×).    an ablation showing the artifacts caused by removing the local coordinate
                                                                           transformation.
                                 Neural Cellular Automata: From Cells to Pixels · 9

Target  Ours  Baseline  Without  LPPN without No Coordinate

        (NCA + LPPN) (NCA + Linear) Perceptual Loss Input Coords Transformation

Fig. 13. Ablations for "Growing a Morphology" experiment. Linear readout (no LPPN) suffers from low quality; removing LPIPS increases blur; removing
coordinate input or sin/cos encoding introduces blob-like or patch artifacts (best viewed digitally after zooming).

Target  Ours  Baseline  Without Multi Without Pseudo + Auto

        (NCA + LPPN) (NCA + Linear) Scale Loss Target Textures Correlation Loss

Fig. 14. Results and ablations for "PBR Texture Synthesis" experiment. NCA+LPPN significantly improves the quality over the baseline; removing multi-scale
loss hurts global coherence and removing pseudo targets breaks cross-map alignment; adding auto-correlation loss improves long-range geometric structure.
10 · Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, and Sabine Süsstrunk

References                                                                                 Vincent Sitzmann, Julien N.P. Martel, Alexander W. Bergman, David B. Lindell, and

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea                 Gordon Wetzstein. 2020. Implicit Neural Representations with Periodic Activation
   Vedaldi. 2014. Describing textures in the wild. In Proceedings of the IEEE conference
    on computer vision and pattern recognition. 3606­3613.                                 Functions. In Proc. NeurIPS.

Kurt W Fleischer, David H Laidlaw, Bena L Currin, and Alan H Barr. 1995. Cellular          Xavier Snelgrove. 2017. High-resolution multi-scale neural texture synthesis. In
    texture generation. In Proceedings of the 22nd annual conference on Computer graphics
    and interactive techniques. 239­248.                                                   SIGGRAPH Asia 2017 Technical Briefs. 1­4.

Leon Gatys, Alexander S Ecker, and Matthias Bethge. 2015. Texture synthesis using          Kenneth O. Stanley. 2007. Compositional pattern producing networks: A novel abstrac-
    convolutional neural networks. Advances in Neural Information Processing Systems
    28 (2015).                                                                             tion of development. Genetic Programming and Evolvable Machines 8, 2 (June 2007),

William Gilpin. 2019. Cellular automata as convolutional neural networks. Physical         131­162. doi:10.1007/s10710-007-9028-8
    Review E 100, 3 (2019), 032402.
                                                                                           Shyam Sudhakaran, Djordje Grbic, Siyan Li, Adam Katona, Elias Najarro, Claire Glanois,
Stéphane Gobron and Norishige Chiba. 1999. 3D surface cellular automata and their
    applications. The Journal of Visualization and Computer Animation 10, 3 (1999),        and Sebastian Risi. 2021. Growing 3D Artefacts and Functional Machines with Neural
   143­158.
                                                                                           Cellular Automata. In 2021 Conference on Artificial Life. https://arxiv.org/abs/2103.
Nicolas Gonthier, Yann Gousseau, and Saïd Ladjal. 2022. High-resolution neural texture
    synthesis with long-range constraints. Journal of Mathematical Imaging and Vision      08737
    64, 5 (2022), 478­492.
                                                                                           Shyam Sudhakaran, Elias Najarro, and Sebastian Risi. 2022. Goal-Guided Neural
Etienne Guichard, Felix Reimers, Mia Kvalsund, Mikkel Lepperød, and Stefano Nichele.
    2025. ARC-NCA: Towards Developmental Solutions to the Abstraction and Reason-          Cellular Automata: Learning to Control Self-Organising Systems. arXiv preprint
    ing Corpus. arXiv preprint arXiv:2505.08778 (2025).
                                                                                           arXiv:2205.06806 (2022).
John Kalkhof, Camila González, and Anirban Mukhopadhyay. 2023. Med-NCA: Robust
    and Lightweight Segmentation with Neural Cellular Automata. In International           Mattie Tesfaldet, Derek Nowrouzezahrai, and Christopher Pal. 2022. Attention-based
    Conference on Information Processing in Medical Imaging. Springer, 705­716.
                                                                                           Neural Cellular Automata. arXiv preprint arXiv:2211.01233 (2022).
John Kalkhof, Arlene Kühn, Yannik Frisch, and Anirban Mukhopadhyay. 2024.
    Frequency-time diffusion with neural cellular automata. arXiv preprint                 AM Turing. 1952. The Chemical Basis of Morphogenesis. Philosophical Transactions of
    arXiv:2401.06291 (2024).
                                                                                           the Royal Society of London Series B 237, 641 (1952), 37­72.
Nicholas Kolkin, Jason Salavon, and Gregory Shakhnarovich. 2019. Style transfer
    by relaxed optimal transport and self-similarity. In Proceedings of the IEEE/CVF       Greg Turk. 1991. Generating textures on arbitrary surfaces using reaction-diffusion.
    Conference on Computer Vision and Pattern Recognition. 10051­10060.
                                                                                           Acm Siggraph Computer Graphics 25, 4 (1991), 289­298.
Maria Larsson, Hodaka Yamaguchi, Ehsan Pajouheshgar, I-Chao Shen, Kenji Tojo,
    Chia-Ming Chang, Lars Hansson, Olof Broman, Takashi Ijiri, Ariel Shamir, Wenzel        Dongqing Wang, Ehsan Pajouheshgar, Yitao Xu, Tong Zhang, and Sabine Süsstrunk.
   Jakob, and Takeo Igarashi. 2025. The Mokume Dataset and Inverse Modeling of
    Solid Wood Textures. ACM Transactions on Graphics 44, 4 (Aug. 2025), 18 pages.         2025. Volumetric Temporal Texture Synthesis for Smoke Stylization using Neural
    doi:10.1145/3730874
                                                                                           Cellular Automata. arXiv preprint arXiv:2502.09631 (2025).
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ra-
    mamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields       Yitao Xu, Ehsan Pajouheshgar, and Sabine Süsstrunk. 2024a.                Emer-
    for View Synthesis. In European Conference on Computer Vision. 405­421.
                                                                                           gent Dynamics in Neural Cellular Automata (Artificial Life Confer-
Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin. 2020.
    Growing Neural Cellular Automata. Distill (2020). doi:10.23915/distill.00023           ence Proceedings, Vol. ALIFE 2024: Proceedings of the 2024 Artificial
    https://distill.pub/2020/growing-ca.
                                                                                           Life Conference). 96.         arXiv:https://direct.mit.edu/isal/proceedings-
John von Neumann. 1966. Theory of self-reproducing automata. Math. Comp. 21 (1966),
   745.                                                                                    pdf/isal2024/36/96/2461079/isal_a_00744.pdf doi:10.1162/isal_a_00744

Eyvind Niklasson, Alexander Mordvintsev, Ettore Randazzo, and Michael Levin. 2021.         Yitao Xu, Tong Zhang, and Sabine Susstrunk. 2024b. AdaNCA: Neural Cellular Automata
    Self-organising textures. Distill 6, 2 (2021), e00027­003.
                                                                                           As Adaptors For More Robust Vision Transformer. In The Thirty-eighth Annual
Maximilian Otte, Quentin Delfosse, Johannes Czech, and Kristian Kersting. 2021. Gen-
    erative adversarial neural cellular automata. arXiv preprint arXiv:2108.04328 (2021).  Conference on Neural Information Processing Systems.

Ehsan Pajouheshgar, Yitao Xu, Alexander Mordvintsev, Eyvind Niklasson, Tong Zhang,         Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The
    and Sabine Süsstrunk. 2024b. Mesh Neural Cellular Automata. ACM Trans. Graph.
   (2024). doi:10.1145/3658127                                                             unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of

Ehsan Pajouheshgar, Yitao Xu, and Sabine Süsstrunk. 2024a. NoiseNCA: Noisy                 the IEEE Conference on Computer Vision and Pattern Recognition. 586­595.
    Seed Improves Spatio-Temporal Continuity of Neural Cellular Automata (Ar-
    tificial Life Conference Proceedings, Vol. ALIFE 2024: Proceedings of the 2024
    Artificial Life Conference). 57. arXiv:https://direct.mit.edu/isal/proceedings-
    pdf/isal2024/36/57/2461193/isal_a_00785.pdf doi:10.1162/isal_a_00785

Ehsan Pajouheshgar, Yitao Xu, Tong Zhang, and Sabine Süsstrunk. 2023. DyNCA: Real-
    time Dynamic Texture Synthesis Using Neural Cellular Automata. In Proceedings of
    the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20742­20751.

Rasmus Berg Palm, Miguel González-Duque, Shyam Sudhakaran, and Sebastian Risi.
    2022. Variational neural cellular automata. arXiv preprint arXiv:2201.12360 (2022).

Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Love-
    grove. 2019. Deepsdf: Learning continuous signed distance functions for shape
    representation. In Proceedings of the IEEE/CVF conference on computer vision and
    pattern recognition. 165­174.

Ettore Randazzo, Alexander Mordvintsev, Eyvind Niklasson, Michael Levin, and Sam
    Greydanus. 2020. Self-classifying MNIST Digits. Distill (2020). doi:10.23915/distill.
    00027.002 https://distill.pub/2020/selforg/mnist.

Mark Sandler, Andrey Zhmoginov, Liangcheng Luo, Alexander Mordvintsev, Ettore
    Randazzo, et al. 2020. Image segmentation via cellular automata. arXiv preprint
    arXiv:2008.04965 (2020).

Omry Sendik and Daniel Cohen-Or. 2017. Deep correlations for texture synthesis. ACM
    Transactions on Graphics (ToG) 36, 5 (2017), 1­15.

Karen Simonyan and Andrew Zisserman. 2015. Very Deep Convolutional Networks
    for Large-Scale Image Recognition. In 3rd International Conference on Learning
    Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track
    Proceedings, Yoshua Bengio and Yann LeCun (Eds.). http://arxiv.org/abs/1409.1556
                                                                                                        Neural Cellular Automata: From Cells to Pixels · 11

A Additional Experiment Details                                          distortions and allows us to precompute barycentric coordinates and
                                                                         face indices for a fixed patch size, avoiding repeated rasterization
A.1 Common Training Recipe                                               during training. For all paper results and for evaluation on arbitrary
                                                                         meshes, we revert to a standard rasterizer with a perspective camera.
Across all experiments, we follow standard NCA training practices.       More details are provided in Appendix C.
We maintain a checkpoint pool of intermediate NCA states to en-
courage long-term stability and expose the model to diverse rollouts;    B Mesh Local Coordinates: Sorting and Analytical CDF
we use a pool size of 512 for all experiments and 1024 for morphol-          Remapping
ogy growth. We use stochastic per-cell updates with probability 0.5
and sample rollout lengths uniformly from the ranges in Table 1.         For mesh texture synthesis, each sampling point p lies in a triangle
Every 32 training iterations, we reset the first batch element to the    face and admits barycentric coordinates (p) = (1, 2, 3) with
seed state (all zeros in all experiments except morphology growth,         0 and   = 1. Directly feeding  to the LPPN is problematic
where the seed is a single active cell at the center). We include an     because adjacent faces may list their vertices in different orders,
overflow regularizer to keep states bounded, | - clip(, -1, 1)|, with    causing discontinuities across shared edges. We therefore (i) sort
weight 100 in all experiments, and normalize gradients of the NCA        the barycentric coordinates and (ii) remap each sorted component
parameters for stability (we do not normalize LPPN gradients). We        to obtain a comparable dynamic range across dimensions.
use an initial learning rate of 10-3 with a step-decay schedule (factor
0.3 every 1000 iterations), unless noted otherwise. Following the        B.1 Sorting for 0 continuity across triangle edges
grafting scheme of Pajouheshgar et al. [2024b], within each experi-      Let
ment we first train a single model from random initialization on an
arbitrary target and use its converged NCA and LPPN weights to                 = max(1, 2, 3),  = min(1, 2, 3),  = 1 -  - , (5)
initialize all other runs in that experiment.
                                                                         so that      are the sorted barycentric coordinates. Along a
A.2 Morphology Growth Training Schedule                                  shared edge, the two nonzero barycentric weights swap between
                                                                         the two incident faces, but sorting makes the ordered tuple (, , )
We train morphology growth for 20k iterations. To reduce sensi-          identical on both sides; hence the local coordinate field becomes 0
tivity to poor local minima, we periodically flush the checkpoint        across triangle boundaries (non-differentiability occurs only on a
pool by replacing all stored states with the seed and simultaneously     measure-zero set where two weights are equal).
restarting the learning-rate schedule; we do this every 4000 iter-
ations. In practice, perceptual quality saturates for many targets       B.2 Distribution of sorted barycentric coordinates under
after 8000 iterations, while a subset continues to improve later in            uniform sampling
training.
                                                                         To equalize the dynamic range of (, , ), we apply a monotone
A.3 Auto-correlation Settings for PBR Textures
                                                                         remapping based on their analytic distributions under uniform sam-
We enable the auto-correlation regularizer only for textures with
strong geometric structure. In our experiments, this term is more        pling within a triangle. For an equilateral triangle, a uniformly
sensitive to hyperparameters than the base multi-scale OT loss: we
typically use weights in the range 10­100 and, for some targets,         sampled point has barycentric coordinates distributed uniformly
extend training beyond the default 3k iterations (up to 20k) to reach
convergence. We apply the auto-correlation term only at the coarsest     over the 2-simplex, i.e., (1, 2, 3)  Dirichlet(1, 1, 1). Equivalently,
scale of the multi-scale texture loss, which we found sufficient in      the joint density of (1, 2) over {1, 2  0, 1 +2  1} is constant,
practice.
                                                                         so probabilities reduce to area ratios.
A.4 Mesh Textures: Coordinate Transform and
       Training-Time Projection                                              We   will  repeatedly      use   the  following  identities  (for           [0,  1  ]  ):
                                                                                                                                                              2
Barycentric coordinate preprocessing. On meshes, the natural
local coordinates within a triangle are barycentric weights, but                                           Pr(1 > ) = (1 - )2,                                      (6)
these are order-dependent because adjacent faces may index vertices
differently. To obtain a consistent coordinate field across triangle                             Pr(1 > , 2 > ) = (1 - 2)2,                                         (7)
boundaries, we sort the barycentric coordinates at each sampling
point (largestsmallest) and then apply a lightweight remapping                          Pr(1 > , 2 > , 3 > ) = (1 - 3)2                   (     1  )  .             (8)
that balances the dynamic range across components. We provide                                                                                   3
the full derivation of this remapping under a uniform-sampling
assumption in Appendix B.                                                  Maximum  = max  . The support of  is [1/3, 1]. For  
                                                                         [1/3, 1/2], we compute Pr(  ) via inclusion­exclusion:
   Cached Lambert projections for training. During training on
the icosphere, we accelerate loss evaluation by replacing perspec-       Pr(  ) = 1 - 3 Pr(1 > ) + 3 Pr(1 > , 2 > ) = (3 - 1)2. (9)
tive rasterization with cached Lambert (equal-area) projections of
random spherical patches. This reduces curvature and perspective         For   [1/2, 1], two components cannot both exceed , hence

                                                                                        Pr(  ) = 1 - 3 Pr(1 > ) = 1 - 3(1 - )2.                                  (10)

                                                                         Differentiating yields a piecewise-linear density with a single peak

                                                                         at    =  1  .
                                                                                  2

                                                                             Minimum  = min  . The support of  is [0, 1/3]. For   [0, 1/3],

                                                                         the event {   } is the shrunken simplex {  ,   = 1},

                                                                         obtained       by  the  shift     =    -    and  a  uniform  scaling      by    (1   -     3 ) .

                                                                                                         
12 · Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, and Sabine Süsstrunk

Since area scales quadratically, we obtain                                                        radius. We then apply the inverse LAEA mapping to convert each
     Pr(  ) = (1 - 3)2  Pr(  ) = 1 - (1 - 3)2. (11)                                               grid point to a unit direction on the sphere. (We use standard LAEA
                                                                                                  formulas via a projection library.)

   Middle component . The support of  is [0, 1/2]. The event { >                                  C.2 Caching barycentric coordinates and face indices
 } is equivalent to "at least two components exceed ". For  
[0, 1/3], the triple intersection is possible, so                                                 Given unit directions {d  } from the sphere center, we intersect
                                                                                                  rays r() =  d  with the icosphere to obtain, for each pixel, the
   Pr( > ) = Pr(at least two  > )                                                           (12)  hit triangle index and barycentric coordinates within that triangle.
                                                                                                  In practice we compute these intersections once for a fixed set of
             = 3 Pr(1 > , 2 > ) - 2 Pr(1 > , 2 > , 3 > ) (13)                                     random patch centers and cache the results:

             = 3(1 - 2)2 - 2(1 - 3)2.                                                       (14)

For   [1/3, 1/2], the triple event is impossible and Pr( > ) =                                                  ×  ×  ×3,             × ×  ×3,

3(1 - 2)2. This yields a continuous, piecewise-quadratic CDF (and                                 faces     N              bary    R            (18)

a  piecewise-linear       density)  with  mode      at    =     1  .                              where  is the number of cached patches, each corresponding to a
                                                                3                                 different viewing direction. During training, we simply sample a sub-
                                                                                                  set of cached patches per iteration and use the stored (faces, bary)
B.3 CDF remapping to uniform coordinates                                                          to interpolate per-vertex NCA states to an image grid. Because the
                                                                                                  cached tensors are constants, this is fully differentiable with respect
The above derivations imply that each of the sorted coordinates                                   to the vertex features (and LPPN parameters) while avoiding re-
follows an exact triangular distribution:                                                         peated rasterization. We use the Lambert cache only during training
                                                                                                  on the icosphere to speed up loss evaluation and reduce projection
     Tri  1  ,  1,  1  ,      Tri      0,  1  ,  1  ,             Tri       0,  1  ,  0  ,  (15)  distortions. For all paper visualizations and for testing on general
          3         2                      2     3                              3                 meshes, we revert to a standard rasterizer with a perspective camera,
                                                                                                  since the cached projection is sphere-specific and does not directly
where Tri(, , ) denotes a triangular distribution on [,  ] with                                   apply to arbitrary mesh geometry.
mode .

   We use the inverse CDF transform to map each component to a
uniform-distribution coordinate. For   Tri(, , ), the CDF is

                               0,                                      < ,

                               
                               
                                       ( - )2                            < ,                      D Interactive Web Demo
                                                                          ,
                                                                                                  We provide an interactive WebGL demo at cells2pixels.github.io
                                                                                                  that runs our trained models fully on the GPU inside the browser.
   Tri (;              , )  =    (  -   ) (      -    ,                                     (16)  The demo is implemented with SwissGL, a lightweight wrapper
                                                    )                                             around WebGL2 designed to minimize boilerplate for managing
                                                                                                  GLSL programs, textures, and framebuffers.

                    ,                     ( - )2                                                     All three demo modes share the same high-level execution pattern.
                                          - )( - )                                                Each animation frame alternates between (1) an NCA step shader
                               1    -  (                     ,                                    that updates the cell state and (2) an LPPN shader that decodes
                                                                                                  the current state into the displayed output. For efficiency, the NCA
                                                                                                  shader fuses perception (local neighborhood aggregation) and the
                                                                                                  two-layer update MLP into a single fragment shader program. Model
                                                                                                  parameters are stored in buffers/textures or uniforms (loaded once
                               1,                                                                 per target), and the NCA state is stored in floating-point textures
                                                                       >  .                       with ping­pong updates.

                                                                                                     In all modes, users can directly perturb the NCA state by interact-
                                                                                                  ing with the canvas. A brush tool either erases local state (setting
We then rescale to [-1, 1]:                                                                       cells/voxels to zero) or inserts a seed (in the morphology demo). A
                                                                                                  speed control adjusts how many NCA steps are executed per ren-
                           = 2 Tri (; , , ) - 1.                                            (17)  dered frame. A scale slider controls the LPPN evaluation resolution:
                                                                                                  although all models are trained with an LPPN upscaling factor of
Applying this to (, , ) yields a continuous local coordinate em-                                  8×, the demo defaults to smaller scales to maintain real-time perfor-
                                                                                                  mance on mobile devices (we use 4× for the morphology and PBR
bedding with balanced dynamic range across the three components.                                  demos, and 2× for the volumetric demo).
The equilateral-triangle assumption is a close match for our training
                                                                                                     When the Transition option is enabled, switching between targets
setup: icosphere faces are near-equilateral, and rasterized samples                               does not reset the current state; instead, the newly selected model
                                                                                                  continues evolving the existing state. This exposes cross-model mor-
are approximately uniform within each face. We therefore use the                                  phing behavior (similar to the parameter-switch morphing shown
analytic CDFs above during training and inference.                                                in our experiments), and allows users to explore attractor changes
                                                                                                  interactively.
C Lambert Patch Projection for Fast Icosphere Training

For mesh texture supervision, we need to evaluate a 2D image-
space loss on signals defined over the sphere. Using a perspective
rasterizer during training introduces (i) perspective foreshortening
and (ii) curvature-induced distortions. Since MeshNCA is trained
on a fixed icosphere [Pajouheshgar et al. 2024b], we can instead
precompute a set of distortion-minimized surface patches and reuse
them throughout training.

C.1 Lambert azimuthal equal-area patches

We parameterize a local spherical patch using the Lambert azimuthal
equal-area (LAEA) projection, which preserves area and therefore
yields approximately uniform sampling density over the patch. For
each cached view, we sample a patch center (latitude/longitude) and
define a regular  ×  grid in the projection plane over a bounded
                                                                        Neural Cellular Automata: From Cells to Pixels · 13

Fig. 15. Interactive web demo for three of our experiments: 2D PBR textures, morphology growth, and 3D volumetric textures.

D.1 Demo modes                                                          texture maps, along with the corresponding renderings in Figures 17-
                                                                        18-19. Each target consists of 9 channels grouped into three maps:
Figure 15 shows the UI of our interactive web demo, including the       Albedo (RGB), Normal (RGB), and HRA (height, roughness, ambient
three modes, target selection panel, and controls for NCA steps per     occlusion). For visualization, we render the texture maps using
frame, LPPN scale, state editing, and model transitions.                a simple PBR shader implemented in PyTorch, including parallax
                                                                        occlusion mapping to account for the height channel.
   PBR textures (2D). The LPPN outputs 9 channels corresponding
to PBR maps (albedo, normal, height, roughness, and ambient oc-            Overall, the synthesized maps closely match the targets, and
clusion). For visualization, we render these maps with a simple PBR     the rendered appearance remains faithful despite training only on
shader and a single point light source. The height channel is addi-     per-map supervision (i.e., without any loss on the final rendered
tionally used to displace the mesh geometry. The demo also includes     image). In some cases, we observe mild patch-like artifacts in regions
an implementation of parallax occlusion mapping that is disabled to     with strong specular response, where small local deviations in the
keep to UI simple. Targets highlighted with a green bounding box        predicted maps can be amplified by the renderer; this is consistent
indicate models trained with auto-correlation supervision enabled.      with the absence of direct supervision on the composed rendering.
                                                                        We leave incorporating a rendering-aware objective (and related
   Morphology (2D). The LPPN directly produces an RGBA image            strategies for reducing such artifacts) to future work.
from the current NCA state, which is displayed without additional
shading. Users can erase regions to test regeneration, or insert a
seed to restart local growth.

   Volumetric textures (3D). We render a learned density field
using a NeRF-style ray-marching shader. For each pixel, we march
along the camera ray through the canonical cube and repeatedly
query the LPPN at sampled 3D locations (using local coordinates
and interpolated NCA state), then composite color and density with
standard front-to-back alpha integration. This enables real-time ex-
ploration of animated volumetric textures, with interactive camera
control and direct state editing.

D.2 Targets

Morphology targets are collected from a publicly available PNG
repository, https://pngimg.com and personal photos. PBR texture
targets use the dataset compiled by Pajouheshgar et al. [2024b]. Volu-
metric texture targets are drawn from three sources: texture images
from DTD [Cimpoi et al. 2014], the textures used in Pajouheshgar
et al. [2023], and additional synthetic texture targets generated via
text-to-image tools.

E More Results

Figure 16 shows more results for the PBR texture synthesis experi-
ments.

   We provide additional qualitative results for the 2D PBR texture
experiment by visualizing the full set of target and synthesized
14 · Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, and Sabine Süsstrunk

Target  Ours  Baseline  Without Multi Without Pseudo + Auto

        (NCA + LPPN) (NCA + Linear) Scale Loss Target Textures Correlation Loss

Fig. 16. Results and ablations for "PBR Texture Synthesis" experiment. NCA+LPPN significantly improves the quality over the baseline; removing multi-scale
loss hurts global coherence and removing pseudo targets breaks cross-map alignment; adding auto-correlation loss improves long-range geometric structure.
                                  Neural Cellular Automata: From Cells to Pixels · 15

             Albedo  Normal  HRA  Rendered

Target

Synthesized

Target

Synthesized

Target

Synthesized

             Fig. 17. Visualization of all texture maps (target and synthesized) for the PBR texture experiment.
16 · Ehsan Pajouheshgar, Yitao Xu, Ali Abbasi, Alexander Mordvintsev, Wenzel Jakob, and Sabine Süsstrunk

             Albedo  Normal  HRA                                                                          Rendered

Target

Synthesized

Target

Synthesized

Target

Synthesized

             Fig. 18. Visualization of all texture maps (target and synthesized) for the PBR texture experiment.
                                  Neural Cellular Automata: From Cells to Pixels · 17

             Albedo  Normal  HRA  Rendered

Target

Synthesized

Target

Synthesized

Target

Synthesized

             Fig. 19. Visualization of all texture maps (target and synthesized) for the PBR texture experiment.
