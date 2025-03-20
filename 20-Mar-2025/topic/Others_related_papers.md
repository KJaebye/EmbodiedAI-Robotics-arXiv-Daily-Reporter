# CCDP: Composition of Conditional Diffusion Policies with Guided Sampling 

**Title (ZH)**: CCDP：带有引导采样的条件扩散策略的组合 

**Authors**: Amirreza Razmjoo, Sylvain Calinon, Michael Gienger, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15386)  

**Abstract**: Imitation Learning offers a promising approach to learn directly from data without requiring explicit models, simulations, or detailed task definitions. During inference, actions are sampled from the learned distribution and executed on the robot. However, sampled actions may fail for various reasons, and simply repeating the sampling step until a successful action is obtained can be inefficient. In this work, we propose an enhanced sampling strategy that refines the sampling distribution to avoid previously unsuccessful actions. We demonstrate that by solely utilizing data from successful demonstrations, our method can infer recovery actions without the need for additional exploratory behavior or a high-level controller. Furthermore, we leverage the concept of diffusion model decomposition to break down the primary problem (which may require long-horizon history to manage failures) into multiple smaller, more manageable sub-problems in learning, data collection, and inference, thereby enabling the system to adapt to variable failure counts. Our approach yields a low-level controller that dynamically adjusts its sampling space to improve efficiency when prior samples fall short. We validate our method across several tasks, including door opening with unknown directions, object manipulation, and button-searching scenarios, demonstrating that our approach outperforms traditional baselines. 

**Abstract (ZH)**: 模仿学习提供了一种有前途的方法，可以直接从数据中学习，而无需显式模型、模拟或详细任务定义。在推理过程中，动作从学习得到的分布中采样并在机器人上执行。然而，采样动作可能因各种原因失败，简单地重复采样步骤直到获得成功动作的执行可能是低效的。在本工作中，我们提出了一种改进的采样策略，通过细化采样分布以避免之前不成功的动作。我们通过仅利用成功演示的数据，可以推断出恢复动作，而无需额外的探索行为或高级控制器。此外，我们利用扩散模型分解的概念，将主要问题分解为多个更小、更易管理的子问题，从而在学习、数据收集和推理过程中使系统能够适应变化的失败次数。我们的方法生成了一个低级控制器，该控制器能根据先前样本的不足动态调整其采样空间，提高效率。我们在多个任务中验证了我们的方法，包括未知方向的门开启、物体操作和按钮搜索场景，结果显示我们的方法优于传统基线。 

---
# HAD-Gen: Human-like and Diverse Driving Behavior Modeling for Controllable Scenario Generation 

**Title (ZH)**: HAD-Gen: 类人类的多样化驾驶行为建模以实现可控场景生成 

**Authors**: Cheng Wang, Lingxin Kong, Massimiliano Tamborski, Stefano V. Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2503.15049)  

**Abstract**: Simulation-based testing has emerged as an essential tool for verifying and validating autonomous vehicles (AVs). However, contemporary methodologies, such as deterministic and imitation learning-based driver models, struggle to capture the variability of human-like driving behavior. Given these challenges, we propose HAD-Gen, a general framework for realistic traffic scenario generation that simulates diverse human-like driving behaviors. The framework first clusters the vehicle trajectory data into different driving styles according to safety features. It then employs maximum entropy inverse reinforcement learning on each of the clusters to learn the reward function corresponding to each driving style. Using these reward functions, the method integrates offline reinforcement learning pre-training and multi-agent reinforcement learning algorithms to obtain general and robust driving policies. Multi-perspective simulation results show that our proposed scenario generation framework can simulate diverse, human-like driving behaviors with strong generalization capability. The proposed framework achieves a 90.96% goal-reaching rate, an off-road rate of 2.08%, and a collision rate of 6.91% in the generalization test, outperforming prior approaches by over 20% in goal-reaching performance. The source code is released at this https URL. 

**Abstract (ZH)**: 基于仿真测试的自动驾驶汽车验证与验证方法：一种通用的现实交通场景生成框架(HAD-Gen) 

---
# DRoPE: Directional Rotary Position Embedding for Efficient Agent Interaction Modeling 

**Title (ZH)**: DRoPE: 方向旋转位置嵌入用于高效代理交互建模 

**Authors**: Jianbo Zhao, Taiyu Ban, Zhihao Liu, Hangning Zhou, Xiyang Wang, Qibin Zhou, Hailong Qin, Mu Yang, Lei Liu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15029)  

**Abstract**: Accurate and efficient modeling of agent interactions is essential for trajectory generation, the core of autonomous driving systems. Existing methods, scene-centric, agent-centric, and query-centric frameworks, each present distinct advantages and drawbacks, creating an impossible triangle among accuracy, computational time, and memory efficiency. To break this limitation, we propose Directional Rotary Position Embedding (DRoPE), a novel adaptation of Rotary Position Embedding (RoPE), originally developed in natural language processing. Unlike traditional relative position embedding (RPE), which introduces significant space complexity, RoPE efficiently encodes relative positions without explicitly increasing complexity but faces inherent limitations in handling angular information due to periodicity. DRoPE overcomes this limitation by introducing a uniform identity scalar into RoPE's 2D rotary transformation, aligning rotation angles with realistic agent headings to naturally encode relative angular information. We theoretically analyze DRoPE's correctness and efficiency, demonstrating its capability to simultaneously optimize trajectory generation accuracy, time complexity, and space complexity. Empirical evaluations compared with various state-of-the-art trajectory generation models, confirm DRoPE's good performance and significantly reduced space complexity, indicating both theoretical soundness and practical effectiveness. The video documentation is available at this https URL. 

**Abstract (ZH)**: 准确高效的代理交互建模对于轨迹生成至关重要，轨迹生成是自主驾驶系统的核心。现有的场景中心、代理中心和查询中心框架各有优势与不足，造成了准确度、计算时间和内存效率之间的不可能三角。为打破这一限制，我们提出了一种新的RoPE扩展——方向旋转位置嵌入（DRoPE），该方法通过在RoPE的2D旋转变换中引入均匀的标识标量，解决了因周期性带来的角度信息处理限制问题，从而自然地编码相对角度信息。我们从理论上分析了DRoPE的正确性和效率，证明了其同时优化轨迹生成准确度、计算时间和空间复杂度的能力。与多种最新的轨迹生成模型的实证比较结果证实了DRoPE在性能和显著降低空间复杂度方面的优势，体现了其理论上的坚实基础和实际的有效性。视频文档可参见该连接。 

---
# Speed Optimization Algorithm based on Deterministic Markov Decision Process for Automated Highway Merge 

**Title (ZH)**: 基于确定性马尔可夫决策过程的高速公路自动汇入速度优化算法 

**Authors**: Takeru Goto, Kosuke Toda, Takayasu Kumano  

**Link**: [PDF](https://arxiv.org/pdf/2503.14899)  

**Abstract**: This study presents a robust optimization algorithm for automated highway merge. The merging scenario is one of the challenging scenes in automated driving, because it requires adjusting ego vehicle's speed to match other vehicles before reaching the end point. Then, we model the speed planning problem as a deterministic Markov decision process. The proposed scheme is able to compute each state value of the process and reliably derive the optimal sequence of actions. In our approach, we adopt jerk as the action of the process to prevent a sudden change of acceleration. However, since this expands the state space, we also consider ways to achieve a real-time operation. We compared our scheme with a simple algorithm with the Intelligent Driver Model. We not only evaluated the scheme in a simulation environment but also conduct a real world testing. 

**Abstract (ZH)**: 一种用于自动高速路合并的鲁棒优化算法 

---
# Curiosity-Diffuser: Curiosity Guide Diffusion Models for Reliability 

**Title (ZH)**: 好奇心弥散器：好奇心引导的扩散模型以提升可靠性 

**Authors**: Zihao Liu, Xing Liu, Yizhai Zhang, Zhengxiong Liu, Panfeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14833)  

**Abstract**: One of the bottlenecks in robotic intelligence is the instability of neural network models, which, unlike control models, lack a well-defined convergence domain and stability. This leads to risks when applying intelligence in the physical world. Specifically, imitation policy based on neural network may generate hallucinations, leading to inaccurate behaviors that impact the safety of real-world applications. To address this issue, this paper proposes the Curiosity-Diffuser, aimed at guiding the conditional diffusion model to generate trajectories with lower curiosity, thereby improving the reliability of policy. The core idea is to use a Random Network Distillation (RND) curiosity module to assess whether the model's behavior aligns with the training data, and then minimize curiosity by classifier guidance diffusion to reduce overgeneralization during inference. Additionally, we propose a computationally efficient metric for evaluating the reliability of the policy, measuring the similarity between the generated behaviors and the training dataset, to facilitate research about reliability learning. Finally, simulation verify the effectiveness and applicability of the proposed method to a variety of scenarios, showing that Curiosity-Diffuser significantly improves task performance and produces behaviors that are more similar to the training data. The code for this work is available at: this http URL 

**Abstract (ZH)**: 一种用于提高神经网络模型稳定性以解决类崖问题的方法：Curiosity-Diffuser及其在增强政策可靠性和泛化能力中的应用 

---
# These Magic Moments: Differentiable Uncertainty Quantification of Radiance Field Models 

**Title (ZH)**: 这些魔幻时刻：辐射场模型的可微不确定性量化 

**Authors**: Parker Ewen, Hao Chen, Seth Isaacson, Joey Wilson, Katherine A. Skinner, Ram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2503.14665)  

**Abstract**: This paper introduces a novel approach to uncertainty quantification for radiance fields by leveraging higher-order moments of the rendering equation. Uncertainty quantification is crucial for downstream tasks including view planning and scene understanding, where safety and robustness are paramount. However, the high dimensionality and complexity of radiance fields pose significant challenges for uncertainty quantification, limiting the use of these uncertainty quantification methods in high-speed decision-making. We demonstrate that the probabilistic nature of the rendering process enables efficient and differentiable computation of higher-order moments for radiance field outputs, including color, depth, and semantic predictions. Our method outperforms existing radiance field uncertainty estimation techniques while offering a more direct, computationally efficient, and differentiable formulation without the need for this http URL uncertainty quantification, we also illustrate the utility of our approach in downstream applications such as next-best-view (NBV) selection and active ray sampling for neural radiance field training. Extensive experiments on synthetic and real-world scenes confirm the efficacy of our approach, which achieves state-of-the-art performance while maintaining simplicity. 

**Abstract (ZH)**: 本文通过利用渲染方程的高阶矩，提出了一种用于辐射场不确定性量化的新方法。不确定性量化对于包括视点规划和场景理解在内的下游任务至关重要，尤其是在安全性和鲁棒性要求高的场合。然而，辐射场的高维性和复杂性极大地阻碍了不确定性量化的实现，限制了这些方法在高速决策中的应用。我们证明，渲染过程的概率性质使得可以直接、高效且可微地计算辐射场输出（包括颜色、深度和语义预测）的高阶矩。我们的方法在不依赖于上述方法的情况下，优于现有的辐射场不确定性估计技术，提供了一种更直接、计算效率更高且可微的表述形式。此外，我们还展示了我们的方法在下游应用中的实用性，如最佳视图选择和神经辐射场训练中的主动光线采样。广泛的合成场景和真实场景实验表明，该方法具有最先进的性能且保持了简单性。 

---
# SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization 

**Title (ZH)**: SuperPC：点云完成、上采样、去噪和着色的单步扩散模型 

**Authors**: Yi Du, Zhipeng Zhao, Shaoshu Su, Sharath Golluri, Haoze Zheng, Runmao Yao, Chen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14558)  

**Abstract**: Point cloud (PC) processing tasks-such as completion, upsampling, denoising, and colorization-are crucial in applications like autonomous driving and 3D reconstruction. Despite substantial advancements, prior approaches often address each of these tasks independently, with separate models focused on individual issues. However, this isolated approach fails to account for the fact that defects like incompleteness, low resolution, noise, and lack of color frequently coexist, with each defect influencing and correlating with the others. Simply applying these models sequentially can lead to error accumulation from each model, along with increased computational costs. To address these challenges, we introduce SuperPC, the first unified diffusion model capable of concurrently handling all four tasks. Our approach employs a three-level-conditioned diffusion framework, enhanced by a novel spatial-mix-fusion strategy, to leverage the correlations among these four defects for simultaneous, efficient processing. We show that SuperPC outperforms the state-of-the-art specialized models as well as their combination on all four individual tasks. 

**Abstract (ZH)**: SuperPC：统一处理点云完成、上采样、去噪和颜色化任务的扩散模型 

---
# Generating Causal Explanations of Vehicular Agent Behavioural Interactions with Learnt Reward Profiles 

**Title (ZH)**: 基于学习到的奖励配置文件生成vehicles代理行为交互的因果解释 

**Authors**: Rhys Howard, Nick Hawes, Lars Kunze  

**Link**: [PDF](https://arxiv.org/pdf/2503.14557)  

**Abstract**: Transparency and explainability are important features that responsible autonomous vehicles should possess, particularly when interacting with humans, and causal reasoning offers a strong basis to provide these qualities. However, even if one assumes agents act to maximise some concept of reward, it is difficult to make accurate causal inferences of agent planning without capturing what is of importance to the agent. Thus our work aims to learn a weighting of reward metrics for agents such that explanations for agent interactions can be causally inferred. We validate our approach quantitatively and qualitatively across three real-world driving datasets, demonstrating a functional improvement over previous methods and competitive performance across evaluation metrics. 

**Abstract (ZH)**: 负责任的自动驾驶车辆在与人类交互时，透明度和可解释性是重要的特征，因果推理为此提供了强有力的 basis。然而，即使假设代理行为是为了最大化某种奖励概念，如果不捕捉代理认为重要的内容，也难以做出准确的因果推断。因此，我们的工作旨在学习代理的奖励度量权重，以便能够因果推理代理交互的解释。我们通过跨三个实际驾驶数据集的定量和定性验证，展示了相对于先前方法的功能改进，并且在评估指标上具有竞争力的性能。 

---
# Value Profiles for Encoding Human Variation 

**Title (ZH)**: 人类变异的特征表示方法 

**Authors**: Taylor Sorensen, Pushkar Mishra, Roma Patel, Michael Henry Tessler, Michiel Bakker, Georgina Evans, Iason Gabriel, Noah Goodman, Verena Rieser  

**Link**: [PDF](https://arxiv.org/pdf/2503.15484)  

**Abstract**: Modelling human variation in rating tasks is crucial for enabling AI systems for personalization, pluralistic model alignment, and computational social science. We propose representing individuals using value profiles -- natural language descriptions of underlying values compressed from in-context demonstrations -- along with a steerable decoder model to estimate ratings conditioned on a value profile or other rater information. To measure the predictive information in rater representations, we introduce an information-theoretic methodology. We find that demonstrations contain the most information, followed by value profiles and then demographics. However, value profiles offer advantages in terms of scrutability, interpretability, and steerability due to their compressed natural language format. Value profiles effectively compress the useful information from demonstrations (>70% information preservation). Furthermore, clustering value profiles to identify similarly behaving individuals better explains rater variation than the most predictive demographic groupings. Going beyond test set performance, we show that the decoder models interpretably change ratings according to semantic profile differences, are well-calibrated, and can help explain instance-level disagreement by simulating an annotator population. These results demonstrate that value profiles offer novel, predictive ways to describe individual variation beyond demographics or group information. 

**Abstract (ZH)**: 基于价值概况的人类变异建模对于实现个性化AI系统、多元模型对齐以及计算社会科学研究至关重要。我们提出使用价值概况——从情境演示中压缩而成的价值自然语言描述来表示个体，并结合可控解码模型，在给定价值概况或其他评分者信息的情况下估计评分。为了衡量评分者表示中的预测信息量，我们引入了一种信息论方法。研究表明，演示包含最多的信 息，其次是价值概况，然后是人口统计学信息。然而，价值概况因其压缩的自然语言格式而在可核查性、可解释性和可控性方面具有优势。价值概况有效地压缩了演示中的有用信息（>70%的信息保留）。此外，通过聚类价值概况来识别具有类似行为的个体，比最具预测性的群体分组更好地解释了评分者差异。超越测试集性能，我们展示了解码模型根据语义概况差异可解释地改变评分，并且校准良好，可以模拟注释员群体来解释实例级别的分歧。这些结果表明，价值概况提供了超越人口统计学或群体信息的新颖、可预测的方式以描述个体差异。 

---
# Dynamic Bi-Elman Attention Networks (DBEAN): Dual-Directional Context-Aware Representation Learning for Enhanced Text Classification 

**Title (ZH)**: 动态双方向上下文感知注意力网络（DBEAN）：增强文本分类的双向上下文表示学习 

**Authors**: ZhengLin Lai, MengYao Liao, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15469)  

**Abstract**: Text classification, a fundamental task in natural language processing (NLP), aims to categorize textual data into predefined labels. Traditional methods struggled with complex linguistic structures and semantic dependencies. The advent of deep learning, particularly recurrent neural networks (RNNs) and Transformer-based models, has significantly advanced the field by enabling nuanced feature extraction and context-aware predictions. Despite improvements, existing models exhibit limitations in balancing interpretability, computational efficiency, and long-range contextual understanding. This paper proposes the Dynamic Bidirectional Elman with Attention Network (DBEAN), which integrates bidirectional temporal modelling with self-attention mechanisms. DBEAN dynamically assigns weights to critical segments of input, improving contextual representation while maintaining computational efficiency. 

**Abstract (ZH)**: 动态双向Elman注意力网络：结合自注意力机制的双向时间建模 

---
# Di$\mathtt{[M]}$O: Distilling Masked Diffusion Models into One-step Generator 

**Title (ZH)**: Di$\mathtt{[M]}$O: 将掩码扩散模型提炼为一步生成器 

**Authors**: Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2503.15457)  

**Abstract**: Masked Diffusion Models (MDMs) have emerged as a powerful generative modeling technique. Despite their remarkable results, they typically suffer from slow inference with several steps. In this paper, we propose Di$\mathtt{[M]}$O, a novel approach that distills masked diffusion models into a one-step generator. Di$\mathtt{[M]}$O addresses two key challenges: (1) the intractability of using intermediate-step information for one-step generation, which we solve through token-level distribution matching that optimizes model output logits by an 'on-policy framework' with the help of an auxiliary model; and (2) the lack of entropy in the initial distribution, which we address through a token initialization strategy that injects randomness while maintaining similarity to teacher training distribution. We show Di$\mathtt{[M]}$O's effectiveness on both class-conditional and text-conditional image generation, impressively achieving performance competitive to multi-step teacher outputs while drastically reducing inference time. To our knowledge, we are the first to successfully achieve one-step distillation of masked diffusion models and the first to apply discrete distillation to text-to-image generation, opening new paths for efficient generative modeling. 

**Abstract (ZH)**: Di$\mathtt{[M]}$O: 一步生成的masked扩散模型蒸馏 

---
# VenusFactory: A Unified Platform for Protein Engineering Data Retrieval and Language Model Fine-Tuning 

**Title (ZH)**: 金星工厂：蛋白质工程数据检索与语言模型微调的统一平台 

**Authors**: Yang Tan, Chen Liu, Jingyuan Gao, Banghao Wu, Mingchen Li, Ruilin Wang, Lingrong Zhang, Huiqun Yu, Guisheng Fan, Liang Hong, Bingxin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.15438)  

**Abstract**: Natural language processing (NLP) has significantly influenced scientific domains beyond human language, including protein engineering, where pre-trained protein language models (PLMs) have demonstrated remarkable success. However, interdisciplinary adoption remains limited due to challenges in data collection, task benchmarking, and application. This work presents VenusFactory, a versatile engine that integrates biological data retrieval, standardized task benchmarking, and modular fine-tuning of PLMs. VenusFactory supports both computer science and biology communities with choices of both a command-line execution and a Gradio-based no-code interface, integrating $40+$ protein-related datasets and $40+$ popular PLMs. All implementations are open-sourced on this https URL. 

**Abstract (ZH)**: 自然语言处理（NLP）超越人类语言对科学领域产生了显著影响，包括蛋白质工程，其中预训练蛋白质语言模型（PLMs）取得了显著成功。然而，由于数据收集、任务基准测试和应用方面的挑战，跨学科应用仍然有限。本工作提出了VenusFactory，这是一个多功能引擎，集成了生物数据检索、标准化任务基准测试和PLMs模块化微调。VenusFactory同时支持计算机科学和生物学社区，提供命令行执行和Gradio基于的无代码界面，整合了40多个蛋白质相关数据集和40多个流行PLMs。所有实现均已开源，链接为：https://github.com/VenusFactory/VenusFactory。 

---
# An extensive simulation study evaluating the interaction of resampling techniques across multiple causal discovery contexts 

**Title (ZH)**: 基于多次因果发现场景下重采样技术交互作用的广泛模拟研究 

**Authors**: Ritwick Banerjee, Bryan Andrews, Erich Kummerfeld  

**Link**: [PDF](https://arxiv.org/pdf/2503.15436)  

**Abstract**: Despite the accelerating presence of exploratory causal analysis in modern science and medicine, the available non-experimental methods for validating causal models are not well characterized. One of the most popular methods is to evaluate the stability of model features after resampling the data, similar to resampling methods for estimating confidence intervals in statistics. Many aspects of this approach have received little to no attention, however, such as whether the choice of resampling method should depend on the sample size, algorithms being used, or algorithm tuning parameters. We present theoretical results proving that certain resampling methods closely emulate the assignment of specific values to algorithm tuning parameters. We also report the results of extensive simulation experiments, which verify the theoretical result and provide substantial data to aid researchers in further characterizing resampling in the context of causal discovery analysis. Together, the theoretical work and simulation results provide specific guidance on how resampling methods and tuning parameters should be selected in practice. 

**Abstract (ZH)**: 尽管探索性因果分析在现代科学和医学中的应用日益加速，现有的非实验性因果模型验证方法尚未得到充分character化。最流行的方法之一是通过重采样数据来评估模型特征的稳定性，类似于统计学中估计置信区间的方法。然而，这种方法的许多方面，例如重采样方法是否应依赖于样本大小、所使用的算法或算法调参参数，尚未得到足够的关注。我们提出了理论结果，证明某些重采样方法可以近似地模拟对算法调参参数赋值的过程。我们还报告了大量模拟实验的结果，验证了理论结果并提供了大量数据，以帮助研究人员进一步在因果发现分析的背景下characterize重采样方法。结合理论工作和模拟结果，我们为如何在实践中选择重采样方法和调参参数提供了具体指导。 

---
# Automated Processing of eXplainable Artificial Intelligence Outputs in Deep Learning Models for Fault Diagnostics of Large Infrastructures 

**Title (ZH)**: 深度学习模型中可解释人工智能输出的自动化处理在大型基础设施故障诊断中的应用 

**Authors**: Giovanni Floreale, Piero Baraldi, Enrico Zio, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2503.15415)  

**Abstract**: Deep Learning (DL) models processing images to recognize the health state of large infrastructure components can exhibit biases and rely on non-causal shortcuts. eXplainable Artificial Intelligence (XAI) can address these issues but manually analyzing explanations generated by XAI techniques is time-consuming and prone to errors. This work proposes a novel framework that combines post-hoc explanations with semi-supervised learning to automatically identify anomalous explanations that deviate from those of correctly classified images and may therefore indicate model abnormal behaviors. This significantly reduces the workload for maintenance decision-makers, who only need to manually reclassify images flagged as having anomalous explanations. The proposed framework is applied to drone-collected images of insulator shells for power grid infrastructure monitoring, considering two different Convolutional Neural Networks (CNNs), GradCAM explanations and Deep Semi-Supervised Anomaly Detection. The average classification accuracy on two faulty classes is improved by 8% and maintenance operators are required to manually reclassify only 15% of the images. We compare the proposed framework with a state-of-the-art approach based on the faithfulness metric: the experimental results obtained demonstrate that the proposed framework consistently achieves F_1 scores larger than those of the faithfulness-based approach. Additionally, the proposed framework successfully identifies correct classifications that result from non-causal shortcuts, such as the presence of ID tags printed on insulator shells. 

**Abstract (ZH)**: 基于后验解释与半监督学习的解释异常自动识别框架：以电力网格基础设施监测中的绝缘子壳体无人机图像为例 

---
# Towards efficient keyword spotting using spike-based time difference encoders 

**Title (ZH)**: 基于脉冲时差编码的高效关键词识别方法 

**Authors**: Alejandro Pequeño-Zurro, Lyes Khacef, Stefano Panzeri, Elisabetta Chicca  

**Link**: [PDF](https://arxiv.org/pdf/2503.15402)  

**Abstract**: Keyword spotting in edge devices is becoming increasingly important as voice-activated assistants are widely used. However, its deployment is often limited by the extreme low-power constraints of the target embedded systems. Here, we explore the Temporal Difference Encoder (TDE) performance in keyword spotting. This recent neuron model encodes the time difference in instantaneous frequency and spike count to perform efficient keyword spotting with neuromorphic processors. We use the TIdigits dataset of spoken digits with a formant decomposition and rate-based encoding into spikes. We compare three Spiking Neural Networks (SNNs) architectures to learn and classify spatio-temporal signals. The proposed SNN architectures are made of three layers with variation in its hidden layer composed of either (1) feedforward TDE, (2) feedforward Current-Based Leaky Integrate-and-Fire (CuBa-LIF), or (3) recurrent CuBa-LIF neurons. We first show that the spike trains of the frequency-converted spoken digits have a large amount of information in the temporal domain, reinforcing the importance of better exploiting temporal encoding for such a task. We then train the three SNNs with the same number of synaptic weights to quantify and compare their performance based on the accuracy and synaptic operations. The resulting accuracy of the feedforward TDE network (89%) is higher than the feedforward CuBa-LIF network (71%) and close to the recurrent CuBa-LIF network (91%). However, the feedforward TDE-based network performs 92% fewer synaptic operations than the recurrent CuBa-LIF network with the same amount of synapses. In addition, the results of the TDE network are highly interpretable and correlated with the frequency and timescale features of the spoken keywords in the dataset. Our findings suggest that the TDE is a promising neuron model for scalable event-driven processing of spatio-temporal patterns. 

**Abstract (ZH)**: 边缘设备中关键词识别在语音激活助手广泛应用的背景下变得越来越重要。然而，其部署往往受限于目标嵌入系统极端的低功耗约束。本文探讨了时差编码器（TDE）在关键词识别中的性能。这种最近的神经元模型通过编码瞬时频率和脉冲计数之间的时差，在神经形态处理器上执行高效的关键词识别。我们使用经过共振分解并基于速率编码成脉冲的TIdigits语音数字数据集。我们将三种脉冲神经网络（SNN）架构用于学习和分类时空信号，并提出了具有变化隐藏层的三层结构，隐藏层由（1）前馈TDE、（2）前馈电流基漏式积分-放电（CuBa-LIF）或（3）递归CuBa-LIF神经元组成。我们首先展示频率转换的语音数字脉冲列在时间域中包含大量信息，强调了更好地利用时间编码对于此类任务的重要性。随后，我们训练了三个具有相同突触权重数量的SNN，以量化并比较其基于准确性和突触操作量的表现。前馈TDE网络的准确率为89%，高于前馈CuBa-LIF网络的71%，接近递归CuBa-LIF网络的91%。然而，前馈TDE网络在相同数量的突触下执行的突触操作少了92%。此外，TDE网络的结果高度可解释，并与数据集中语音关键词的频率和时间尺度特征相关。我们的发现表明，TDE是可用于时空模式可扩展事件驱动处理的有前途的神经元模型。 

---
# Optimizing Decomposition for Optimal Claim Verification 

**Title (ZH)**: 优化分解以实现最优索赔验证 

**Authors**: Yining Lu, Noah Ziems, Hy Dang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15354)  

**Abstract**: Current research on the \textit{Decompose-Then-Verify} paradigm for evaluating the factuality of long-form text typically treats decomposition and verification in isolation, overlooking their interactions and potential misalignment. We find that existing decomposition policies, typically hand-crafted demonstrations, do not align well with downstream verifiers in terms of atomicity -- a novel metric quantifying information density -- leading to suboptimal verification results. We formulate finding the optimal decomposition policy for optimal verification as a bilevel optimization problem. To approximate a solution for this strongly NP-hard problem, we propose dynamic decomposition, a reinforcement learning framework that leverages verifier feedback to learn a policy for dynamically decomposing claims to verifier-preferred atomicity. Experimental results show that dynamic decomposition outperforms existing decomposition policies, improving verification confidence by 0.07 and accuracy by 0.12 (on a 0-1 scale) on average across varying verifiers, datasets, and atomcities of input claims. 

**Abstract (ZH)**: 当前关于\textit{分解后验证}范式评价长篇文本事实性的研究通常将分解和验证分离，忽视了它们之间的互动和潜在的不一致。我们发现现有的分解策略，通常为手工构建的示例，与下游验证器在原子性方面（原子性是衡量信息密度的新型指标）不一致，导致验证结果次优。我们将寻找最优分解策略以实现最优验证建模为一个双层优化问题。为了近似解决这一强NP难问题，我们提出了一种动力分解框架，该框架利用验证器的反馈学习一种动力分解策略，以达到验证器偏好化的原子性。实验结果表明，动力分解在所有验证器、不同数据集和输入声明原子性差异的情况下，优于现有的分解策略，平均将验证置信度提高0.07和准确性提高0.12（在0-1尺度上）。 

---
# CoE: Chain-of-Explanation via Automatic Visual Concept Circuit Description and Polysemanticity Quantification 

**Title (ZH)**: CoE:基于自动视觉概念电路描述和多义性量化的过程解释 

**Authors**: Wenlong Yu, Qilong Wang, Chuang Liu, Dong Li, Qinghua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15234)  

**Abstract**: Explainability is a critical factor influencing the wide deployment of deep vision models (DVMs). Concept-based post-hoc explanation methods can provide both global and local insights into model decisions. However, current methods in this field face challenges in that they are inflexible to automatically construct accurate and sufficient linguistic explanations for global concepts and local circuits. Particularly, the intrinsic polysemanticity in semantic Visual Concepts (VCs) impedes the interpretability of concepts and DVMs, which is underestimated severely. In this paper, we propose a Chain-of-Explanation (CoE) approach to address these issues. Specifically, CoE automates the decoding and description of VCs to construct global concept explanation datasets. Further, to alleviate the effect of polysemanticity on model explainability, we design a concept polysemanticity disentanglement and filtering mechanism to distinguish the most contextually relevant concept atoms. Besides, a Concept Polysemanticity Entropy (CPE), as a measure of model interpretability, is formulated to quantify the degree of concept uncertainty. The modeling of deterministic concepts is upgraded to uncertain concept atom distributions. Finally, CoE automatically enables linguistic local explanations of the decision-making process of DVMs by tracing the concept circuit. GPT-4o and human-based experiments demonstrate the effectiveness of CPE and the superiority of CoE, achieving an average absolute improvement of 36% in terms of explainability scores. 

**Abstract (ZH)**: 基于概念的解释链（CoE）方法：提高深度视觉模型解释性的新途径 

---
# Multi-Agent Actor-Critic with Harmonic Annealing Pruning for Dynamic Spectrum Access Systems 

**Title (ZH)**: 多代理actor-critic算法结合谐波退火修剪方法在动态频谱访问系统中的应用 

**Authors**: George Stamatelis, Angelos-Nikolaos Kanatas, George C. Alexandropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.15172)  

**Abstract**: Multi-Agent Deep Reinforcement Learning (MADRL) has emerged as a powerful tool for optimizing decentralized decision-making systems in complex settings, such as Dynamic Spectrum Access (DSA). However, deploying deep learning models on resource-constrained edge devices remains challenging due to their high computational cost. To address this challenge, in this paper, we present a novel sparse recurrent MARL framework integrating gradual neural network pruning into the independent actor global critic paradigm. Additionally, we introduce a harmonic annealing sparsity scheduler, which achieves comparable, and in certain cases superior, performance to standard linear and polynomial pruning schedulers at large sparsities. Our experimental investigation demonstrates that the proposed DSA framework can discover superior policies, under diverse training conditions, outperforming conventional DSA, MADRL baselines, and state-of-the-art pruning techniques. 

**Abstract (ZH)**: 多代理深度强化学习（MADRL）在网络资源受限的动态频谱访问（DSA）等复杂环境中优化去中心化决策系统方面已成为一种强有力的工具。然而，在资源受限的边缘设备上部署深度学习模型仍然具有挑战性，原因在于其高计算成本。为解决这一挑战，本文提出了一种新颖的稀疏递归多代理强化学习框架，该框架将渐进神经网络剪枝技术集成到独立演员全局批评家范式中。此外，我们引入了一种谐波退火稀疏调度器，在高稀疏性下能够实现与标准线性和多项式剪枝调度器相当，甚至更优的性能。实验研究证明，所提出的DSA框架在各种训练条件下能够发现更优策略，超越了传统的DSA、MADRL基线和最先进的剪枝技术。 

---
# A Foundational Theory for Decentralized Sensory Learning 

**Title (ZH)**: 去中心化感官学习的基础理论 

**Authors**: Linus Mårtensson, Jonas M.D. Enander, Udaya B. Rongala, Henrik Jörntell  

**Link**: [PDF](https://arxiv.org/pdf/2503.15130)  

**Abstract**: In both neuroscience and artificial intelligence, popular functional frameworks and neural network formulations operate by making use of extrinsic error measurements and global learning algorithms. Through a set of conjectures based on evolutionary insights on the origin of cellular adaptive mechanisms, we reinterpret the core meaning of sensory signals to allow the brain to be interpreted as a negative feedback control system, and show how this could lead to local learning algorithms without the need for global error correction metrics. Thereby, a sufficiently good minima in sensory activity can be the complete reward signal of the network, as well as being both necessary and sufficient for biological learning to arise. We show that this method of learning was likely already present in the earliest unicellular life forms on earth. We show evidence that the same principle holds and scales to multicellular organisms where it in addition can lead to division of labour between cells. Available evidence shows that the evolution of the nervous system likely was an adaptation to more effectively communicate intercellular signals to support such division of labour. We therefore propose that the same learning principle that evolved already in the earliest unicellular life forms, i.e. negative feedback control of externally and internally generated sensor signals, has simply been scaled up to become a fundament of the learning we see in biological brains today. We illustrate diverse biological settings, from the earliest unicellular organisms to humans, where this operational principle appears to be a plausible interpretation of the meaning of sensor signals in biology, and how this relates to current neuroscientific theories and findings. 

**Abstract (ZH)**: 在神经科学和人工智能中，流行的功能性框架和神经网络形式化方法通过利用外在误差测量和全局学习算法来运作。通过基于细胞适应机制起源的进化洞察的一系列猜想，我们重新解释了感官信号的核心意义，允许大脑被解释为一个负反馈控制系统，并展示了如何这可以导致无需全局误差矫正度量的局部学习算法。因此，感官活动中的足够好的极小值可以成为网络的完整奖励信号，同时也是生物学习出现的必要和充分条件。我们证明，在地球最早期的单细胞生物中很可能就已经存在这种学习方式。我们还展示了同样的原理在多细胞生物中适用并可扩展，而且在这种情况下，它还可能导致细胞之间的劳动分工。现有证据表明，神经系统的进化很可能是为了更有效地传递细胞间信号，以支持这种劳动分工。因此，我们提出，在最早期的单细胞生物中已经进化出的负反馈控制外部和内部生成的传感器信号的学习原理，只是在今天生物大脑中的学习中得到了放大。我们展示了从最早期的单细胞生物到人类的多样化的生物设置，其中这种操作原理似乎是生物学中传感器信号意义的一种合理解释，并探讨了它与当前神经科学理论和发现之间的关系。 

---
# Diffusion-Based Forecasting for Uncertainty-Aware Model Predictive Control 

**Title (ZH)**: 基于扩散的预测方法以实现不确定性感知的模型预测控制 

**Authors**: Stelios Zarifis, Ioannis Kordonis, Petros Maragos  

**Link**: [PDF](https://arxiv.org/pdf/2503.15095)  

**Abstract**: We propose Diffusion-Informed Model Predictive Control (D-I MPC), a generic framework for uncertainty-aware prediction and decision-making in partially observable stochastic systems by integrating diffusion-based time series forecasting models in Model Predictive Control algorithms. In our approach, a diffusion-based time series forecasting model is used to probabilistically estimate the evolution of the system's stochastic components. These forecasts are then incorporated into MPC algorithms to estimate future trajectories and optimize action selection under the uncertainty of the future. We evaluate the framework on the task of energy arbitrage, where a Battery Energy Storage System participates in the day-ahead electricity market of the New York state. Experimental results indicate that our model-based approach with a diffusion-based forecaster significantly outperforms both implementations with classical forecasting methods and model-free reinforcement learning baselines. 

**Abstract (ZH)**: 基于扩散模型的模型预测控制（D-I MPC）：在部分可观测随机系统中实现不确定性意识的预测与决策 

---
# Conjuring Positive Pairs for Efficient Unification of Representation Learning and Image Synthesis 

**Title (ZH)**: 召唤正样本对以高效统一表示学习与图像合成 

**Authors**: Imanol G. Estepa, Jesús M. Rodríguez-de-Vera, Ignacio Sarasúa, Bhalaji Nagarajan, Petia Radeva  

**Link**: [PDF](https://arxiv.org/pdf/2503.15060)  

**Abstract**: While representation learning and generative modeling seek to understand visual data, unifying both domains remains unexplored. Recent Unified Self-Supervised Learning (SSL) methods have started to bridge the gap between both paradigms. However, they rely solely on semantic token reconstruction, which requires an external tokenizer during training -- introducing a significant overhead. In this work, we introduce Sorcen, a novel unified SSL framework, incorporating a synergic Contrastive-Reconstruction objective. Our Contrastive objective, "Echo Contrast", leverages the generative capabilities of Sorcen, eliminating the need for additional image crops or augmentations during training. Sorcen "generates" an echo sample in the semantic token space, forming the contrastive positive pair. Sorcen operates exclusively on precomputed tokens, eliminating the need for an online token transformation during training, thereby significantly reducing computational overhead. Extensive experiments on ImageNet-1k demonstrate that Sorcen outperforms the previous Unified SSL SoTA by 0.4%, 1.48 FID, 1.76%, and 1.53% on linear probing, unconditional image generation, few-shot learning, and transfer learning, respectively, while being 60.8% more efficient. Additionally, Sorcen surpasses previous single-crop MIM SoTA in linear probing and achieves SoTA performance in unconditional image generation, highlighting significant improvements and breakthroughs in Unified SSL models. 

**Abstract (ZH)**: Sorcen：一种新颖的协同对比重建统一自监督学习框架 

---
# Application of linear regression method to the deep reinforcement learning in continuous action cases 

**Title (ZH)**: 线性回归方法在连续动作情况下的深度强化学习应用 

**Authors**: Hisato Komatsu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14976)  

**Abstract**: The linear regression (LR) method offers the advantage that optimal parameters can be calculated relatively easily, although its representation capability is limited than that of the deep learning technique. To improve deep reinforcement learning, the Least Squares Deep Q Network (LS-DQN) method was proposed by Levine et al., which combines Deep Q Network (DQN) with LR method. However, the LS-DQN method assumes that the actions are discrete. In this study, we propose the Double Least Squares Deep Deterministic Policy Gradient (DLS-DDPG) method to address this limitation. This method combines the LR method with the Deep Deterministic Policy Gradient (DDPG) technique, one of the representative deep reinforcement learning algorithms for continuous action cases. Numerical experiments conducted in MuJoCo environments showed that the LR update improved performance at least in some tasks, although there are difficulties such as the inability to make the regularization terms small. 

**Abstract (ZH)**: 基于最小二乘的双层深度确定性策略梯度方法（DLS-DDPG） 

---
# A Semantic and Clean-label Backdoor Attack against Graph Convolutional Networks 

**Title (ZH)**: 面向图卷积网络的语义和干净标签后门攻击 

**Authors**: Jiazhu Dai, Haoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.14922)  

**Abstract**: Graph Convolutional Networks (GCNs) have shown excellent performance in graph-structured tasks such as node classification and graph classification. However, recent research has shown that GCNs are vulnerable to a new type of threat called the backdoor attack, where the adversary can inject a hidden backdoor into the GCNs so that the backdoored model performs well on benign samples, whereas its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. Clean-label backdoor attack and semantic backdoor attack are two new backdoor attacks to Deep Neural Networks (DNNs), they are more imperceptible and have posed new and serious threats. The semantic and clean-label backdoor attack is not fully explored in GCNs. In this paper, we propose a semantic and clean-label backdoor attack against GCNs under the context of graph classification to reveal the existence of this security vulnerability in GCNs. Specifically, SCLBA conducts an importance analysis on graph samples to select one type of node as semantic trigger, which is then inserted into the graph samples to create poisoning samples without changing the labels of the poisoning samples to the attacker-specified target label. We evaluate SCLBA on multiple datasets and the results show that SCLBA can achieve attack success rates close to 99% with poisoning rates of less than 3%, and with almost no impact on the performance of model on benign samples. 

**Abstract (ZH)**: Graph卷积网络中的语义清洁标签后门攻击 

---
# Efficient Personalization of Quantized Diffusion Model without Backpropagation 

**Title (ZH)**: 无需反向传播的量化扩散模型高效个性化 

**Authors**: Hoigi Seo, Wongi Jeong, Kyungryeol Lee, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2503.14868)  

**Abstract**: Diffusion models have shown remarkable performance in image synthesis, but they demand extensive computational and memory resources for training, fine-tuning and inference. Although advanced quantization techniques have successfully minimized memory usage for inference, training and fine-tuning these quantized models still require large memory possibly due to dequantization for accurate computation of gradients and/or backpropagation for gradient-based algorithms. However, memory-efficient fine-tuning is particularly desirable for applications such as personalization that often must be run on edge devices like mobile phones with private data. In this work, we address this challenge by quantizing a diffusion model with personalization via Textual Inversion and by leveraging a zeroth-order optimization on personalization tokens without dequantization so that it does not require gradient and activation storage for backpropagation that consumes considerable memory. Since a gradient estimation using zeroth-order optimization is quite noisy for a single or a few images in personalization, we propose to denoise the estimated gradient by projecting it onto a subspace that is constructed with the past history of the tokens, dubbed Subspace Gradient. In addition, we investigated the influence of text embedding in image generation, leading to our proposed time steps sampling, dubbed Partial Uniform Timestep Sampling for sampling with effective diffusion timesteps. Our method achieves comparable performance to prior methods in image and text alignment scores for personalizing Stable Diffusion with only forward passes while reducing training memory demand up to $8.2\times$. 

**Abstract (ZH)**: 通过Textual Inversion量化个性化扩散模型并利用零阶优化进行内存高效微调 

---
# 1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities 

**Title (ZH)**: 1000层网络用于自监督RL：扩大深度可以实现新的目标达成能力 

**Authors**: Kevin Wang, Ishaan Javali, Michał Bortkiewicz, Tomasz Trzciński, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2503.14858)  

**Abstract**: Scaling up self-supervised learning has driven breakthroughs in language and vision, yet comparable progress has remained elusive in reinforcement learning (RL). In this paper, we study building blocks for self-supervised RL that unlock substantial improvements in scalability, with network depth serving as a critical factor. Whereas most RL papers in recent years have relied on shallow architectures (around 2 - 5 layers), we demonstrate that increasing the depth up to 1024 layers can significantly boost performance. Our experiments are conducted in an unsupervised goal-conditioned setting, where no demonstrations or rewards are provided, so an agent must explore (from scratch) and learn how to maximize the likelihood of reaching commanded goals. Evaluated on simulated locomotion and manipulation tasks, our approach increases performance by $2\times$ - $50\times$. Increasing the model depth not only increases success rates but also qualitatively changes the behaviors learned. 

**Abstract (ZH)**: 自监督强化学习中规模提升的构建块：网络深度的关键作用与显著性能增益 

---
# The CLEF-2025 CheckThat! Lab: Subjectivity, Fact-Checking, Claim Normalization, and Retrieval 

**Title (ZH)**: CLEF-2025 CheckThat! 实验室：主观性、事实核查、论断规范化与检索 

**Authors**: Firoj Alam, Julia Maria Struß, Tanmoy Chakraborty, Stefan Dietze, Salim Hafid, Katerina Korre, Arianna Muti, Preslav Nakov, Federico Ruggeri, Sebastian Schellhammer, Vinay Setty, Megha Sundriyal, Konstantin Todorov, Venktesh V  

**Link**: [PDF](https://arxiv.org/pdf/2503.14828)  

**Abstract**: The CheckThat! lab aims to advance the development of innovative technologies designed to identify and counteract online disinformation and manipulation efforts across various languages and platforms. The first five editions focused on key tasks in the information verification pipeline, including check-worthiness, evidence retrieval and pairing, and verification. Since the 2023 edition, the lab has expanded its scope to address auxiliary tasks that support research and decision-making in verification. In the 2025 edition, the lab revisits core verification tasks while also considering auxiliary challenges. Task 1 focuses on the identification of subjectivity (a follow-up from CheckThat! 2024), Task 2 addresses claim normalization, Task 3 targets fact-checking numerical claims, and Task 4 explores scientific web discourse processing. These tasks present challenging classification and retrieval problems at both the document and span levels, including multilingual settings. 

**Abstract (ZH)**: CheckThat!实验室旨在推进识别和对抗各种语言和平台上的在线虚假信息和操纵努力的创新技术的发展。前五届活动主要关注信息核实管道中的关键任务，包括可核实性、证据检索和配对以及核实。从2023年版开始，实验室扩大了范围，以支持核实研究和决策的辅助任务。在2025年版中，实验室重新审视核心核实任务，同时考虑辅助挑战。任务1关注主观性识别（继承自CheckThat! 2024），任务2处理主张规范化，任务3针对事实核查数值声明，任务4探索科学网络话语处理。这些任务在文档级和短语级上提出了具有挑战性的分类和检索问题，包括多语种设置。 

---
# Learning with Expert Abstractions for Efficient Multi-Task Continuous Control 

**Title (ZH)**: 基于专家抽象的高效多任务连续控制学习 

**Authors**: Jeff Jewett, Sandhya Saisubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2503.14809)  

**Abstract**: Decision-making in complex, continuous multi-task environments is often hindered by the difficulty of obtaining accurate models for planning and the inefficiency of learning purely from trial and error. While precise environment dynamics may be hard to specify, human experts can often provide high-fidelity abstractions that capture the essential high-level structure of a task and user preferences in the target environment. Existing hierarchical approaches often target discrete settings and do not generalize across tasks. We propose a hierarchical reinforcement learning approach that addresses these limitations by dynamically planning over the expert-specified abstraction to generate subgoals to learn a goal-conditioned policy. To overcome the challenges of learning under sparse rewards, we shape the reward based on the optimal state value in the abstract model. This structured decision-making process enhances sample efficiency and facilitates zero-shot generalization. Our empirical evaluation on a suite of procedurally generated continuous control environments demonstrates that our approach outperforms existing hierarchical reinforcement learning methods in terms of sample efficiency, task completion rate, scalability to complex tasks, and generalization to novel scenarios. 

**Abstract (ZH)**: 在复杂连续多任务环境中的决策制定往往受到准确建模规划的难度以及仅通过试错学习的低效性的阻碍。虽然环境动力学可能难以精确描述，但人类专家通常可以提供高保真抽象，捕获任务的基本高层结构和目标环境中的用户偏好。现有层次化方法往往针对离散设置，并且不能在任务之间泛化。我们提出了一种层次化强化学习方法，通过动态规划专家指定的抽象来生成子目标，以学习一个以目标条件化策略。为了克服在稀疏奖励下的学习挑战，我们基于抽象模型中的最优状态值塑造奖励。这种结构化的决策过程提高了样本效率并促进了零样本泛化。我们在一系列程序生成的连续控制环境中的实证评估表明，我们的方法在样本效率、任务完成率、复杂任务的可扩展性以及向新颖场景的泛化方面优于现有的层次化强化学习方法。 

---
# Long Context Modeling with Ranked Memory-Augmented Retrieval 

**Title (ZH)**: 带排名记忆增强检索的长上下文建模 

**Authors**: Ghadir Alselwi, Hao Xue, Shoaib Jameel, Basem Suleiman, Flora D. Salim, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2503.14800)  

**Abstract**: Effective long-term memory management is crucial for language models handling extended contexts. We introduce a novel framework that dynamically ranks memory entries based on relevance. Unlike previous works, our model introduces a novel relevance scoring and a pointwise re-ranking model for key-value embeddings, inspired by learning-to-rank techniques in information retrieval. Enhanced Ranked Memory Augmented Retrieval ERMAR achieves state-of-the-art results on standard benchmarks. 

**Abstract (ZH)**: 一种基于动态相关性排序的记忆管理框架实现语言模型在处理长上下文时的有效长期记忆管理。ERMAR：增强排序记忆扩展检索取得标准基准上的最佳效果。 

---
# RAT: Boosting Misclassification Detection Ability without Extra Data 

**Title (ZH)**: RAT: 在无需额外数据的情况下提升误分类检测能力 

**Authors**: Ge Yan, Tsui-Wei Weng  

**Link**: [PDF](https://arxiv.org/pdf/2503.14783)  

**Abstract**: As deep neural networks(DNN) become increasingly prevalent, particularly in high-stakes areas such as autonomous driving and healthcare, the ability to detect incorrect predictions of models and intervene accordingly becomes crucial for safety. In this work, we investigate the detection of misclassified inputs for image classification models from the lens of adversarial perturbation: we propose to use robust radius (a.k.a. input-space margin) as a confidence metric and design two efficient estimation algorithms, RR-BS and RR-Fast, for misclassification detection. Furthermore, we design a training method called Radius Aware Training (RAT) to boost models' ability to identify mistakes. Extensive experiments show our method could achieve up to 29.3% reduction on AURC and 21.62% reduction in FPR@95TPR, compared with previous methods. 

**Abstract (ZH)**: 随着深度神经网络(DNN)在自动驾驶和医疗等高风险领域中的广泛应用，检测模型的错误预测并及时干预以确保安全变得至关重要。在本工作中，我们从对手扰动的角度研究了图像分类模型的误分类检测：我们提出使用鲁棒半径（即输入空间间隔）作为置信度度量，并设计了两种高效的误分类检测算法RR-BS和RR-Fast。此外，我们提出了一种名为鲁棒半径感知训练（RAT）的训练方法，以提高模型识别错误的能力。大量实验结果显示，与之前的方法相比，我们的方法在AURC上最多可减少29.3%，在FPR@95TPR上减少21.62%。 

---
# Language Independent Named Entity Recognition via Orthogonal Transformation of Word Vectors 

**Title (ZH)**: 基于词向量正交变换的无语言依赖命名实体识别 

**Authors**: Omar E. Rakha, Hazem M. Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2503.14755)  

**Abstract**: Word embeddings have been a key building block for NLP in which models relied heavily on word embeddings in many different tasks. In this paper, a model is proposed based on using Bidirectional LSTM/CRF with word embeddings to perform named entity recognition for any language. This is done by training a model on a source language (English) and transforming word embeddings from the target language into word embeddings of the source language by using an orthogonal linear transformation matrix. Evaluation of the model shows that by training a model on an English dataset the model was capable of detecting named entities in an Arabic dataset without neither training or fine tuning the model on an Arabic language dataset. 

**Abstract (ZH)**: 基于双向LSTM/CRF和词嵌入的语言无关命名实体识别模型 

---
# Bayesian Modeling of Zero-Shot Classifications for Urban Flood Detection 

**Title (ZH)**: 零 shot 分类的贝叶斯建模在城市洪涝检测中的应用 

**Authors**: Matt Franchi, Nikhil Garg, Wendy Ju, Emma Pierson  

**Link**: [PDF](https://arxiv.org/pdf/2503.14754)  

**Abstract**: Street scene datasets, collected from Street View or dashboard cameras, offer a promising means of detecting urban objects and incidents like street flooding. However, a major challenge in using these datasets is their lack of reliable labels: there are myriad types of incidents, many types occur rarely, and ground-truth measures of where incidents occur are lacking. Here, we propose BayFlood, a two-stage approach which circumvents this difficulty. First, we perform zero-shot classification of where incidents occur using a pretrained vision-language model (VLM). Second, we fit a spatial Bayesian model on the VLM classifications. The zero-shot approach avoids the need to annotate large training sets, and the Bayesian model provides frequent desiderata in urban settings - principled measures of uncertainty, smoothing across locations, and incorporation of external data like stormwater accumulation zones. We comprehensively validate this two-stage approach, showing that VLMs provide strong zero-shot signal for floods across multiple cities and time periods, the Bayesian model improves out-of-sample prediction relative to baseline methods, and our inferred flood risk correlates with known external predictors of risk. Having validated our approach, we show it can be used to improve urban flood detection: our analysis reveals 113,738 people who are at high risk of flooding overlooked by current methods, identifies demographic biases in existing methods, and suggests locations for new flood sensors. More broadly, our results showcase how Bayesian modeling of zero-shot LM annotations represents a promising paradigm because it avoids the need to collect large labeled datasets and leverages the power of foundation models while providing the expressiveness and uncertainty quantification of Bayesian models. 

**Abstract (ZH)**: 基于街景数据的BayFlood双阶段方法：规避标签难题，提高城市洪水检测能力 

---
# LipShiFT: A Certifiably Robust Shift-based Vision Transformer 

**Title (ZH)**: LipShiFT: 一种可认证稳健的基于移位的视觉变换器 

**Authors**: Rohan Menon, Nicola Franco, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2503.14751)  

**Abstract**: Deriving tight Lipschitz bounds for transformer-based architectures presents a significant challenge. The large input sizes and high-dimensional attention modules typically prove to be crucial bottlenecks during the training process and leads to sub-optimal results. Our research highlights practical constraints of these methods in vision tasks. We find that Lipschitz-based margin training acts as a strong regularizer while restricting weights in successive layers of the model. Focusing on a Lipschitz continuous variant of the ShiftViT model, we address significant training challenges for transformer-based architectures under norm-constrained input setting. We provide an upper bound estimate for the Lipschitz constants of this model using the $l_2$ norm on common image classification datasets. Ultimately, we demonstrate that our method scales to larger models and advances the state-of-the-art in certified robustness for transformer-based architectures. 

**Abstract (ZH)**: 基于变压器的架构获取紧的利普希茨界是一个重大挑战。大输入尺寸和高维注意力模块通常在训练过程中成为关键瓶颈，导致次优结果。我们的研究强调了这些方法在视觉任务中的实际限制。我们发现基于利普希茨的边际训练作为一种强正则化手段，能够限制模型后续层中的权重。我们专注于ShiftViT模型的连续变体，在范数约束输入设置下，解决了变压器架构的重要训练挑战。我们使用常见的图像分类数据集上的$l_2$范数，提供了该模型的利普希茨常数的上界估计。最终，我们证明了该方法可扩展到更大的模型，并在变压器架构的验证鲁棒性方面取得了最先进的成果。 

---
# DPImageBench: A Unified Benchmark for Differentially Private Image Synthesis 

**Title (ZH)**: DPImageBench: 一体化差分隐私图像合成基准 

**Authors**: Chen Gong, Kecen Li, Zinan Lin, Tianhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14681)  

**Abstract**: Differentially private (DP) image synthesis aims to generate artificial images that retain the properties of sensitive images while protecting the privacy of individual images within the dataset. Despite recent advancements, we find that inconsistent--and sometimes flawed--evaluation protocols have been applied across studies. This not only impedes the understanding of current methods but also hinders future advancements.
To address the issue, this paper introduces DPImageBench for DP image synthesis, with thoughtful design across several dimensions: (1) Methods. We study eleven prominent methods and systematically characterize each based on model architecture, pretraining strategy, and privacy mechanism. (2) Evaluation. We include nine datasets and seven fidelity and utility metrics to thoroughly assess them. Notably, we find that a common practice of selecting downstream classifiers based on the highest accuracy on the sensitive test set not only violates DP but also overestimates the utility scores. DPImageBench corrects for these mistakes. (3) Platform. Despite the methods and evaluation protocols, DPImageBench provides a standardized interface that accommodates current and future implementations within a unified framework. With DPImageBench, we have several noteworthy findings. For example, contrary to the common wisdom that pretraining on public image datasets is usually beneficial, we find that the distributional similarity between pretraining and sensitive images significantly impacts the performance of the synthetic images and does not always yield improvements. In addition, adding noise to low-dimensional features, such as the high-level characteristics of sensitive images, is less affected by the privacy budget compared to adding noise to high-dimensional features, like weight gradients. The former methods perform better than the latter under a low privacy budget. 

**Abstract (ZH)**: 不同隐私保护（DP）图像合成的研究评估标准不一，且有时存在缺陷，这不仅阻碍了当前方法的理解，也妨碍了未来的发展。为了解决这一问题，本文提出了DPImageBench，从多个维度进行了精心设计：方法方面，我们研究了十一种主流方法，并基于模型架构、预训练策略和隐私机制对每种方法进行了系统化描述；评估方面，我们包含了九个数据集和七项保真度与实用性指标，以全面评估这些方法。值得注意的是，我们发现根据敏感测试集上的最高准确性选择下游分类器不仅违反了DP，还高估了实用性得分，DPImageBench纠正了这些错误；平台方面，尽管有方法和评估标准的不同，DPImageBench仍提供了一套标准化接口，适用于当前和未来的统一框架实现。通过使用DPImageBench，我们获得了几个重要发现。例如，与普遍观点相反，我们发现，预训练在公共图像数据集上的效果并不总是最优，预训练和敏感图像之间的分布相似性显著影响合成图像的表现，并不总是带来改进。此外，在低维度特征（如敏感图像的高层次特性）上添加噪声比在高维度特征（如权重梯度）上添加噪声受隐私预算的影响更小，在低隐私预算下前者方法的表现优于后者。 

---
# ConQuer: A Framework for Concept-Based Quiz Generation 

**Title (ZH)**: ConQuer：基于概念的quiz生成框架 

**Authors**: Yicheng Fu, Zikui Wang, Liuxin Yang, Meiqing Huo, Zhongdongming Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.14662)  

**Abstract**: Quizzes play a crucial role in education by reinforcing students' understanding of key concepts and encouraging self-directed exploration. However, compiling high-quality quizzes can be challenging and require deep expertise and insight into specific subject matter. Although LLMs have greatly enhanced the efficiency of quiz generation, concerns remain regarding the quality of these AI-generated quizzes and their educational impact on students. To address these issues, we introduce ConQuer, a concept-based quiz generation framework that leverages external knowledge sources. We employ comprehensive evaluation dimensions to assess the quality of the generated quizzes, using LLMs as judges. Our experiment results demonstrate a 4.8% improvement in evaluation scores and a 77.52% win rate in pairwise comparisons against baseline quiz sets. Ablation studies further underscore the effectiveness of each component in our framework. Code available at this https URL. 

**Abstract (ZH)**: 概念导向的 quiz 生成框架 ConQuer 在利用外部知识源的基础上提升 quiz 质量及教育影响 

---
# Core-Periphery Principle Guided State Space Model for Functional Connectome Classification 

**Title (ZH)**: 由核心-边缘原则引导的空间模型在功能联接组分类中的应用 

**Authors**: Minheng Chen, Xiaowei Yu, Jing Zhang, Tong Chen, Chao Cao, Yan Zhuang, Yanjun Lyu, Lu Zhang, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14655)  

**Abstract**: Understanding the organization of human brain networks has become a central focus in neuroscience, particularly in the study of functional connectivity, which plays a crucial role in diagnosing neurological disorders. Advances in functional magnetic resonance imaging and machine learning techniques have significantly improved brain network analysis. However, traditional machine learning approaches struggle to capture the complex relationships between brain regions, while deep learning methods, particularly Transformer-based models, face computational challenges due to their quadratic complexity in long-sequence modeling. To address these limitations, we propose a Core-Periphery State-Space Model (CP-SSM), an innovative framework for functional connectome classification. Specifically, we introduce Mamba, a selective state-space model with linear complexity, to effectively capture long-range dependencies in functional brain networks. Furthermore, inspired by the core-periphery (CP) organization, a fundamental characteristic of brain networks that enhances efficient information transmission, we design CP-MoE, a CP-guided Mixture-of-Experts that improves the representation learning of brain connectivity patterns. We evaluate CP-SSM on two benchmark fMRI datasets: ABIDE and ADNI. Experimental results demonstrate that CP-SSM surpasses Transformer-based models in classification performance while significantly reducing computational complexity. These findings highlight the effectiveness and efficiency of CP-SSM in modeling brain functional connectivity, offering a promising direction for neuroimaging-based neurological disease diagnosis. 

**Abstract (ZH)**: 理解人类大脑网络的组织已经成为神经科学中的一个核心焦点，特别是在功能性连接的研究中，后者在神经障碍诊断中扮演着至关重要的角色。功能性磁共振成像技术和机器学习方法的进步显著改善了大脑网络分析。然而，传统的机器学习方法难以捕捉大脑区域之间的复杂关系，而基于Transformer的深度学习方法在长时间序列建模中面临着计算上的挑战。为了解决这些局限性，我们提出了一种核心-外围状态空间模型（CP-SSM），这是一种用于功能性连接体分类的创新框架。具体来说，我们引入了一种选择性状态空间模型Mamba，该模型具有线性复杂度，能够有效地捕捉功能性脑网络中的长程依赖关系。此外，受到脑网络中核心-外围（CP）组织这一基本原则的启发，这种组织形式提升了信息传输效率，我们设计了一种CP引导的专家混合模型CP-MoE，以改善脑连接模式的表示学习。我们在两个基准fMRI数据集ABIDE和ADNI上评估了CP-SSM。实验结果表明，CP-SSM在分类性能上超过了基于Transformer的模型，同时显著降低了计算复杂度。这些发现突显了CP-SSM在建模脑功能连接方面的有效性和效率，为神经影像学基础上的神经疾病诊断提供了有前景的方向。 

---
# Reducing False Ventricular Tachycardia Alarms in ICU Settings: A Machine Learning Approach 

**Title (ZH)**: 在ICU环境中减少假性室性心动过速警报：一种机器学习方法 

**Authors**: Grace Funmilayo Farayola, Akinyemi Sadeeq Akintola, Oluwole Fagbohun, Chukwuka Michael Oforgu, Bisola Faith Kayode, Christian Chimezie, Temitope Kadri, Abiola Oludotun, Nelson Ogbeide, Mgbame Michael, Adeseye Ifaturoti, Toyese Oloyede  

**Link**: [PDF](https://arxiv.org/pdf/2503.14621)  

**Abstract**: False arrhythmia alarms in intensive care units (ICUs) are a significant challenge, contributing to alarm fatigue and potentially compromising patient safety. Ventricular tachycardia (VT) alarms are particularly difficult to detect accurately due to their complex nature. This paper presents a machine learning approach to reduce false VT alarms using the VTaC dataset, a benchmark dataset of annotated VT alarms from ICU monitors. We extract time-domain and frequency-domain features from waveform data, preprocess the data, and train deep learning models to classify true and false VT alarms. Our results demonstrate high performance, with ROC-AUC scores exceeding 0.96 across various training configurations. This work highlights the potential of machine learning to improve the accuracy of VT alarm detection in clinical settings. 

**Abstract (ZH)**: ICUs中室性心动过速假警报的机器学习减缓方法：基于VTaC数据集的研究 

---
# PHGNN: A Novel Prompted Hypergraph Neural Network to Diagnose Alzheimer's Disease 

**Title (ZH)**: PHGNN：一种新型提示超图神经网络用于诊断阿尔茨海默病 

**Authors**: Chenyu Liu, Luca Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2503.14577)  

**Abstract**: The accurate diagnosis of Alzheimer's disease (AD) and prognosis of mild cognitive impairment (MCI) conversion are crucial for early intervention. However, existing multimodal methods face several challenges, from the heterogeneity of input data, to underexplored modality interactions, missing data due to patient dropouts, and limited data caused by the time-consuming and costly data collection process. In this paper, we propose a novel Prompted Hypergraph Neural Network (PHGNN) framework that addresses these limitations by integrating hypergraph based learning with prompt learning. Hypergraphs capture higher-order relationships between different modalities, while our prompt learning approach for hypergraphs, adapted from NLP, enables efficient training with limited data. Our model is validated through extensive experiments on the ADNI dataset, outperforming SOTA methods in both AD diagnosis and the prediction of MCI conversion. 

**Abstract (ZH)**: 阿尔茨海默病（AD）的准确诊断和轻度认知 impairment （MCI）向 AD 转变的预后对于早期干预至关重要。现有多种模态方法面临数据异质性、模态交互未充分探索、患者退出导致的数据缺失以及由于耗时且成本高的数据收集过程而引起的数据有限等问题。本文提出了一种新颖的Prompted Hypergraph Neural Network (PHGNN)框架，通过结合超图学习和提示学习来解决这些问题。我们的模型在ADNI数据集上的广泛实验中表现出色，不仅在AD诊断中优于当前最佳方法，还在MCI转换的预测中也表现出色。 

---
# SocialJax: An Evaluation Suite for Multi-agent Reinforcement Learning in Sequential Social Dilemmas 

**Title (ZH)**: SocialJax：在序列社会困境中多智能体强化学习的评估套件 

**Authors**: Zihao Guo, Richard Willis, Shuqing Shi, Tristan Tomilin, Joel Z. Leibo, Yali Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.14576)  

**Abstract**: Social dilemmas pose a significant challenge in the field of multi-agent reinforcement learning (MARL). Melting Pot is an extensive framework designed to evaluate social dilemma environments, providing an evaluation protocol that measures generalization to new social partners across various test scenarios. However, running reinforcement learning algorithms in the official Melting Pot environments demands substantial computational resources. In this paper, we introduce SocialJax, a suite of sequential social dilemma environments implemented in JAX. JAX is a high-performance numerical computing library for Python that enables significant improvements in the operational efficiency of SocialJax on GPUs and TPUs. Our experiments demonstrate that the training pipeline of SocialJax achieves a 50\texttimes{} speedup in real-time performance compared to Melting Pot's RLlib baselines. Additionally, we validate the effectiveness of baseline algorithms within the SocialJax environments. Finally, we use Schelling diagrams to verify the social dilemma properties of these environments, ensuring they accurately capture the dynamics of social dilemmas. 

**Abstract (ZH)**: 多智能体 reinforcement 学习领域的社会困境构成重大挑战。Melting Pot 是一个广泛采用的框架，旨在评估社会困境环境，并提供一种评估协议，该协议衡量在各种测试场景中对新社会伙伴的泛化能力。然而，在官方 Melting Pot 环境中运行 reinforcement 学习算法需要大量计算资源。本文介绍了基于 JAX 实现的一系列顺序社会困境环境——SocialJax。JAX 是一个高性能的 Python 数值计算库，使 SocialJax 在 GPU 和 TPU 上的运行效率显著提高。我们的实验表明，SocialJax 的训练管道在实时性能上比 Melting Pot 的 RLlib 基准速度快 50 倍。此外，我们验证了 SocialJax 环境中基础算法的有效性。最后，我们使用 Schelling 图来验证这些环境的社会困境特性，确保它们能够准确捕捉社会困境的动力学。 

---
# Potential Score Matching: Debiasing Molecular Structure Sampling with Potential Energy Guidance 

**Title (ZH)**: 潜在能评分匹配：以潜在能为导向的分子结构采样去偏差化 

**Authors**: Liya Guo, Zun Wang, Chang Liu, Junzhe Li, Pipi Hu, Yi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14569)  

**Abstract**: The ensemble average of physical properties of molecules is closely related to the distribution of molecular conformations, and sampling such distributions is a fundamental challenge in physics and chemistry. Traditional methods like molecular dynamics (MD) simulations and Markov chain Monte Carlo (MCMC) sampling are commonly used but can be time-consuming and costly. Recently, diffusion models have emerged as efficient alternatives by learning the distribution of training data. Obtaining an unbiased target distribution is still an expensive task, primarily because it requires satisfying ergodicity. To tackle these challenges, we propose Potential Score Matching (PSM), an approach that utilizes the potential energy gradient to guide generative models. PSM does not require exact energy functions and can debias sample distributions even when trained on limited and biased data. Our method outperforms existing state-of-the-art (SOTA) models on the Lennard-Jones (LJ) potential, a commonly used toy model. Furthermore, we extend the evaluation of PSM to high-dimensional problems using the MD17 and MD22 datasets. The results demonstrate that molecular distributions generated by PSM more closely approximate the Boltzmann distribution compared to traditional diffusion models. 

**Abstract (ZH)**: 分子的ensemble平均物理性质与分子构象分布密切相关，而采样这种分布是物理学和化学中的基本挑战。传统的分子动力学（MD）模拟和马尔可夫链蒙特卡洛（MCMC）采样方法常用但耗时且成本高。近年来，通过学习训练数据分布的扩散模型已 emerge 为有效的替代方法。获得无偏目标分布仍然是一个昂贵的任务，主要是因为它要求满足遍历性。为应对这些挑战，我们提出了势能评分匹配（PSM）方法，该方法利用势能梯度引导生成模型。PSM 不需要精确的能量函数，即使在训练数据有限且有偏的情况下也能消除样本分布的偏差。我们的方法在Lennard-Jones（LJ）势能下优于现有最先进的（SOTA）模型，该势能常被用作玩具模型。此外，我们通过MD17和MD22数据集评估了PSM在高维问题上的表现。结果表明，PSM生成的分子分布更接近玻耳兹曼分布，相比传统扩散模型表现出更佳的结果。 

---
# SpecReX: Explainable AI for Raman Spectroscopy 

**Title (ZH)**: SpecReX: 可解释的人工智能在拉曼光谱中的应用 

**Authors**: Nathan Blake, David A. Kelly, Akchunya Chanchal, Sarah Kapllani-Mucaj, Geraint Thomas, Hana Chockler  

**Link**: [PDF](https://arxiv.org/pdf/2503.14567)  

**Abstract**: Raman spectroscopy is becoming more common for medical diagnostics with deep learning models being increasingly used to leverage its full potential. However, the opaque nature of such models and the sensitivity of medical diagnosis together with regulatory requirements necessitate the need for explainable AI tools. We introduce SpecReX, specifically adapted to explaining Raman spectra. SpecReX uses the theory of actual causality to rank causal responsibility in a spectrum, quantified by iteratively refining mutated versions of the spectrum and testing if it retains the original classification. The explanations provided by SpecReX take the form of a responsibility map, highlighting spectral regions most responsible for the model to make a correct classification. To assess the validity of SpecReX, we create increasingly complex simulated spectra, in which a "ground truth" signal is seeded, to train a classifier. We then obtain SpecReX explanations and compare the results with another explainability tool. By using simulated spectra we establish that SpecReX localizes to the known differences between classes, under a number of conditions. This provides a foundation on which we can find the spectral features which differentiate disease classes. This is an important first step in proving the validity of SpecReX. 

**Abstract (ZH)**: 拉曼光谱学在医学诊断中的应用日益增多，深度学习模型的使用使其潜力得到更大发挥。然而，这类模型的黑盒性质、医学诊断的敏感性以及监管要求促使需要可解释的人工智能工具。我们介绍了一种专门用于解释拉曼光谱的SpecReX工具。SpecReX利用实际因果理论对光谱中的因果责任进行排名，通过迭代优化光谱的变异版本并测试其是否保留原始分类来实现。SpecReX提供的解释采取责任图的形式，突出显示对模型正确分类影响最大的光谱区域。为验证SpecReX的有效性，我们创建了越来越复杂的模拟光谱，在其中植入“真实信号”以训练分类器。然后我们使用SpecReX获得解释，并将其结果与另一种可解释性工具进行比较。通过使用模拟光谱，我们发现在多种条件下，SpecReX能够定位到不同类别之间的已知差异，为发现区分疾病类别的光谱特征奠定了基础。这是证明SpecReX有效性的关键一步。 

---
# Effortless Active Labeling for Long-Term Test-Time Adaptation 

**Title (ZH)**: 无努力的主动标注以实现长期测试时自适应 

**Authors**: Guowei Wang, Changxing Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.14564)  

**Abstract**: Long-term test-time adaptation (TTA) is a challenging task due to error accumulation. Recent approaches tackle this issue by actively labeling a small proportion of samples in each batch, yet the annotation burden quickly grows as the batch number increases. In this paper, we investigate how to achieve effortless active labeling so that a maximum of one sample is selected for annotation in each batch. First, we annotate the most valuable sample in each batch based on the single-step optimization perspective in the TTA context. In this scenario, the samples that border between the source- and target-domain data distributions are considered the most feasible for the model to learn in one iteration. Then, we introduce an efficient strategy to identify these samples using feature perturbation. Second, we discover that the gradient magnitudes produced by the annotated and unannotated samples have significant variations. Therefore, we propose balancing their impact on model optimization using two dynamic weights. Extensive experiments on the popular ImageNet-C, -R, -K, -A and PACS databases demonstrate that our approach consistently outperforms state-of-the-art methods with significantly lower annotation costs. 

**Abstract (ZH)**: 长期内存时自适应（TTA）在由于误差累积而成为一个具有挑战性的任务。近期的方法通过在每个批次中积极标注少量样本来解决这一问题，然而随着批次数量的增加，标注负担迅速增长。在本文中，我们探讨了如何实现轻松的主动标注，以便在每个批次中最多只选择一个样本进行标注。首先，我们基于TTA上下文的一步优化视角，标注每个批次中最有价值的样本。在这种情况下，位于源领域和目标领域数据分布之间的样本被认为是模型在一个迭代中学习的最佳选择。然后，我们引入了一种高效的方法来使用特征扰动识别这些样本。其次，我们发现标注样本和未标注样本产生的梯度幅度存在显著差异。因此，我们提出使用两个动态权重来平衡它们对模型优化的影响。在流行的ImageNet-C、-R、-K、-A和PACS数据库上的广泛实验表明，我们的方法在显著降低标注成本的同时，能够持续优于现有最佳方法。 

---
# Workflow for Safe-AI 

**Title (ZH)**: Safe-AI工作流 

**Authors**: Suzana Veljanovska, Hans Dermot Doran  

**Link**: [PDF](https://arxiv.org/pdf/2503.14563)  

**Abstract**: The development and deployment of safe and dependable AI models is crucial in applications where functional safety is a key concern. Given the rapid advancement in AI research and the relative novelty of the safe-AI domain, there is an increasing need for a workflow that balances stability with adaptability. This work proposes a transparent, complete, yet flexible and lightweight workflow that highlights both reliability and qualifiability. The core idea is that the workflow must be qualifiable, which demands the use of qualified tools. Tool qualification is a resource-intensive process, both in terms of time and cost. We therefore place value on a lightweight workflow featuring a minimal number of tools with limited features. The workflow is built upon an extended ONNX model description allowing for validation of AI algorithms from their generation to runtime deployment. This validation is essential to ensure that models are validated before being reliably deployed across different runtimes, particularly in mixed-criticality systems. Keywords-AI workflows, safe-AI, dependable-AI, functional safety, v-model development 

**Abstract (ZH)**: 安全可靠的人工智能模型的发展与部署在功能安全至关重要的应用中至关重要。鉴于人工智能研究的快速进展以及安全人工智能领域的相对新颖性，需要一种平衡稳定性和适应性的工作流。本文提出了一种透明、全面但灵活且轻量级的工作流，强调可靠性和可认证性。核心思想是工作流必须可认证，这要求使用合格的工具。工具认证是一个耗时且耗资的过程。因此，我们重视一种轻量级的工作流，其中包含具有有限功能的最少工具。该工作流建立在扩展的ONNX模型描述之上，允许从生成到运行时部署验证人工智能算法。这种验证对于确保在不同运行时环境中可靠部署模型，特别是在混合关键性系统中，至关重要。关键词-AI工作流，安全人工智能，可靠人工智能，功能安全，V模型开发。 

---
# Squeeze Out Tokens from Sample for Finer-Grained Data Governance 

**Title (ZH)**: 从样本中挤出令牌以实现更细粒度的数据治理 

**Authors**: Weixiong Lin, Chen Ju, Haicheng Wang, Shengchao Hu, Shuai Xiao, Mengting Chen, Yuheng Jiao, Mingshuai Yao, Jinsong Lan, Qingwen Liu, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.14559)  

**Abstract**: Widely observed data scaling laws, in which error falls off as a power of the training size, demonstrate the diminishing returns of unselective data expansion. Hence, data governance is proposed to downsize datasets through pruning non-informative samples. Yet, isolating the impact of a specific sample on overall model performance is challenging, due to the vast computation required for tryout all sample combinations. Current data governors circumvent this complexity by estimating sample contributions through heuristic-derived scalar scores, thereby discarding low-value ones. Despite thorough sample sieving, retained samples contain substantial undesired tokens intrinsically, underscoring the potential for further compression and purification. In this work, we upgrade data governance from a 'sieving' approach to a 'juicing' one. Instead of scanning for least-flawed samples, our dual-branch DataJuicer applies finer-grained intra-sample governance. It squeezes out informative tokens and boosts image-text alignments. Specifically, the vision branch retains salient image patches and extracts relevant object classes, while the text branch incorporates these classes to enhance captions. Consequently, DataJuicer yields more refined datasets through finer-grained governance. Extensive experiments across datasets demonstrate that DataJuicer significantly outperforms existing DataSieve in image-text retrieval, classification, and dense visual reasoning. 

**Abstract (ZH)**: 广泛观察到的数据缩放定律表明，误差随训练数据量的增加呈幂次衰减，这展示了非选择性数据扩增的递减回报。因此，提议通过修剪非信息性样本来减少数据集规模进行数据治理。然而，隔离单个样本对整体模型性能的影响极具挑战性，因为需要进行大量计算来尝试所有样本组合。当前的数据经理通过启发式衍生的标量评分估计样本贡献，从而丢弃低价值的样本。尽管进行了彻底的样本筛选，保留的样本中仍然包含大量内在的不希望出现的标记，这表明进一步压缩和净化的潜力。在此项工作中，我们将数据治理从“筛选”方法升级为“压榨”方法。与其寻找最无瑕的样本，我们的双分支DataJuicer应用更精细粒度的样本内治理。它挤出有价值的信息标记，并增强图像-文本对齐。具体来说，视觉分支保留显著的图像块并提取相关的对象类别，而文本分支将这些类别纳入以增强描述。因此，DataJuicer通过更精细粒度的治理产生更精细的数据集。广泛的数据集实验表明，DataJuicer在图像-文本检索、分类和密集视觉推理中显著优于现有的DataSieve。 

---
# Designing and Deploying AI Models for Sustainable Logistics Optimization: A Case Study on Eco-Efficient Supply Chains in the USA 

**Title (ZH)**: 设计并部署.AI模型以实现可持续物流优化：以美国生态高效供应链案例研究为例 

**Authors**: Reza E Rabbi Shawon, MD Rokibul Hasan, Md Anisur Rahman, Mohamed Ghandri, Iman Ahmed Lamari, Mohammed Kawsar, Rubi Akter  

**Link**: [PDF](https://arxiv.org/pdf/2503.14556)  

**Abstract**: The rapid evolution of Artificial Intelligence (AI) and Machine Learning (ML) has significantly transformed logistics and supply chain management, particularly in the pursuit of sustainability and eco-efficiency. This study explores AI-based methodologies for optimizing logistics operations in the USA, focusing on reducing environmental impact, improving fuel efficiency, and minimizing costs. Key AI applications include predictive analytics for demand forecasting, route optimization through machine learning, and AI-powered fuel efficiency strategies. Various models, such as Linear Regression, XGBoost, Support Vector Machine, and Neural Networks, are applied to real-world logistics datasets to reduce carbon emissions based on logistics operations, optimize travel routes to minimize distance and travel time, and predict future deliveries to plan optimal routes. Other models such as K-Means and DBSCAN are also used to optimize travel routes to minimize distance and travel time for logistics operations. This study utilizes datasets from logistics companies' databases. The study also assesses model performance using metrics such as mean absolute error (MAE), mean squared error (MSE), and R2 score. This study also explores how these models can be deployed to various platforms for real-time logistics and supply chain use. The models are also examined through a thorough case study, highlighting best practices and regulatory frameworks that promote sustainability. The findings demonstrate AI's potential to enhance logistics efficiency, reduce carbon footprints, and contribute to a more resilient and adaptive supply chain ecosystem. 

**Abstract (ZH)**: 人工智能和机器学习的快速演进显著 transforming 物流和供应链管理，特别是在追求可持续性和生态效率方面的努力。本文探讨了基于人工智能的方法论以优化美国的物流操作，重点关注减少环境影响、提高燃料效率和降低成本。关键的人工智能应用包括基于预测分析的需求预测、通过机器学习实现的路线优化以及人工智能驱动的燃料效率策略。应用了线性回归、XGBoost、支持向量机和神经网络等多种模型，基于物流操作数据减少碳排放、优化旅行路线以最小化距离和旅行时间，并预测未来交付以规划最优路线。其他模型如K-均值和DBSCAN也被用于最小化物流操作中的距离和旅行时间以优化旅行路线。本文利用了物流公司的数据库数据集。研究还使用平均绝对误差（MAE）、均方误差（MSE）和R2评分等指标评估模型性能。本文还探讨了这些模型如何部署到各种平台以实现实时的物流和供应链使用。通过详尽的案例研究，本文还探讨了促进可持续性的最佳实践和监管框架。研究发现表明，人工智能有可能提升物流效率、减少碳足迹，并促进更具韧性和适应性的供应链生态系统。 

---
# A Generalist Hanabi Agent 

**Title (ZH)**: 通用型Hanabi智能体 

**Authors**: Arjun V Sudhakar, Hadi Nekoei, Mathieu Reymond, Miao Liu, Janarthanan Rajendran, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2503.14555)  

**Abstract**: Traditional multi-agent reinforcement learning (MARL) systems can develop cooperative strategies through repeated interactions. However, these systems are unable to perform well on any other setting than the one they have been trained on, and struggle to successfully cooperate with unfamiliar collaborators. This is particularly visible in the Hanabi benchmark, a popular 2-to-5 player cooperative card-game which requires complex reasoning and precise assistance to other agents. Current MARL agents for Hanabi can only learn one specific game-setting (e.g., 2-player games), and play with the same algorithmic agents. This is in stark contrast to humans, who can quickly adjust their strategies to work with unfamiliar partners or situations. In this paper, we introduce Recurrent Replay Relevance Distributed DQN (R3D2), a generalist agent for Hanabi, designed to overcome these limitations. We reformulate the task using text, as language has been shown to improve transfer. We then propose a distributed MARL algorithm that copes with the resulting dynamic observation- and action-space. In doing so, our agent is the first that can play all game settings concurrently, and extend strategies learned from one setting to other ones. As a consequence, our agent also demonstrates the ability to collaborate with different algorithmic agents -- agents that are themselves unable to do so. The implementation code is available at: $\href{this https URL}{R3D2-A-Generalist-Hanabi-Agent}$ 

**Abstract (ZH)**: 传统的多代理强化学习（MARL）系统可以通过重复交互发展合作策略。然而，这些系统在训练环境之外的表现不佳，并且难以与不熟悉的合作者成功合作。这一点在Hanabi基准中尤为明显，Hanabi是一个流行的合作纸牌游戏，需要复杂的推理和对其他代理的精确协助。目前的Hanabi MARL代理只能学习一种特定的游戏设置（例如2人游戏），并且使用相同的算法代理进行游戏。这与人类的行为形成鲜明对比，人类能够快速调整策略以适应不熟悉的合作伙伴或情况。在本文中，我们引入了循环回放相关分布式DQN（R3D2），这是一种用于Hanabi的通用代理，旨在克服这些限制。我们通过文本重新定义了任务，因为语言已被证明可以改进迁移。然后，我们提出了一种能够应对由此产生的动态观测空间和动作空间的分布式MARL算法。因此，我们的代理可以同时玩所有游戏设置，并将从一种设置中学到的策略扩展到其他设置。因此，我们的代理还展示了与不同算法代理协作的能力——这些代理本身也无法做到这一点。相关实现代码可在以下链接获得：R3D2-A通用Hanabi代理$\href{this https URL}{R3D2-A-Generalist-Hanabi-Agent}$ 

---
# Novel AI-Based Quantification of Breast Arterial Calcification to Predict Cardiovascular Risk 

**Title (ZH)**: 基于人工智能的新穎乳腺动脉钙化定量方法以预测心血管风险 

**Authors**: Theodorus Dapamede, Aisha Urooj, Vedant Joshi, Gabrielle Gershon, Frank Li, Mohammadreza Chavoshi, Beatrice Brown-Mulry, Rohan Satya Isaac, Aawez Mansuri, Chad Robichaux, Chadi Ayoub, Reza Arsanjani, Laurence Sperling, Judy Gichoya, Marly van Assen, Charles W. ONeill, Imon Banerjee, Hari Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2503.14550)  

**Abstract**: Women are underdiagnosed and undertreated for cardiovascular disease. Automatic quantification of breast arterial calcification on screening mammography can identify women at risk for cardiovascular disease and enable earlier treatment and management of disease. In this retrospective study of 116,135 women from two healthcare systems, a transformer-based neural network quantified BAC severity (no BAC, mild, moderate, and severe) on screening mammograms. Outcomes included major adverse cardiovascular events (MACE) and all-cause mortality. BAC severity was independently associated with MACE after adjusting for cardiovascular risk factors, with increasing hazard ratios from mild (HR 1.18-1.22), moderate (HR 1.38-1.47), to severe BAC (HR 2.03-2.22) across datasets (all p<0.001). This association remained significant across all age groups, with even mild BAC indicating increased risk in women under 50. BAC remained an independent predictor when analyzed alongside ASCVD risk scores, showing significant associations with myocardial infarction, stroke, heart failure, and mortality (all p<0.005). Automated BAC quantification enables opportunistic cardiovascular risk assessment during routine mammography without additional radiation or cost. This approach provides value beyond traditional risk factors, particularly in younger women, offering potential for early CVD risk stratification in the millions of women undergoing annual mammography. 

**Abstract (ZH)**: 女性心血管疾病诊断不足且治疗不足。基于Transformer的神经网络在筛查乳腺X线摄影中自动量化乳腺动脉钙化可以识别心血管疾病风险女性并促进疾病的早期治疗和管理。在两项 healthcare 系统中的116,135名女性的回顾性研究中，变压器基神经网络在筛查乳腺X线摄影中量化了乳腺动脉钙化严重程度（无钙化、轻度、中度和重度）。结果包括主要不良心血管事件（MACE）和全因死亡率。钙化严重程度在调整心血管风险因素后独立与MACE相关，从轻度钙化（HR 1.18-1.22）、中度钙化（HR 1.38-1.47）到重度钙化（HR 2.03-2.22）的危险比在不同数据集中逐渐增加（所有p<0.001）。这种关联在所有年龄组中均具显著性，即使轻度钙化也增加了50岁以下女性的患病风险。在同时分析冠状动脉性心脏病风险评分时，钙化仍是一个独立的预测因子，显示出与心肌梗死、中风、心力衰竭和死亡的显著关联（所有p<0.005）。自动量化乳腺动脉钙化可以在常规乳腺X线摄影中进行机会性心血管风险评估，无需额外辐射或成本。这种方法为年轻女性提供了传统风险因素之外的额外价值，特别是在每年接受乳腺X线摄影的数百万女性中提供早期心血管疾病风险分层的潜力。 

---
# Sampling Decisions 

**Title (ZH)**: 采样决策 

**Authors**: Michael Chertkov, Sungsoo Ahn, Hamidreza Behjoo  

**Link**: [PDF](https://arxiv.org/pdf/2503.14549)  

**Abstract**: In this manuscript we introduce a novel Decision Flow (DF) framework for sampling from a target distribution while incorporating additional guidance from a prior sampler. DF can be viewed as an AI driven algorithmic reincarnation of the Markov Decision Process (MDP) approach in Stochastic Optimal Control. It extends the continuous space, continuous time path Integral Diffusion sampling technique to discrete time and space, while also generalizing the Generative Flow Network framework. In its most basic form, an explicit, Neural Network (NN) free formulation, DF leverages the linear solvability of the the underlying MDP to adjust the transition probabilities of the prior sampler. The resulting Markov Process is expressed as a convolution of the reverse time Green's function of the prior sampling with the target distribution. We illustrate the DF framework through an example of sampling from the Ising model, discuss potential NN based extensions, and outline how DF can enhance guided sampling across various applications. 

**Abstract (ZH)**: 基于先验采样器的附加指导的新型决策流框架 

---
# The Impact of Artificial Intelligence on Emergency Medicine: A Review of Recent Advances 

**Title (ZH)**: 人工智能对急诊医学的影响：近期进展综述 

**Authors**: Gustavo Correia, Victor Alves, Paulo Novais  

**Link**: [PDF](https://arxiv.org/pdf/2503.14546)  

**Abstract**: Artificial Intelligence (AI) is revolutionizing emergency medicine by enhancing diagnostic processes and improving patient outcomes. This article provides a review of the current applications of AI in emergency imaging studies, focusing on the last five years of advancements. AI technologies, particularly machine learning and deep learning, are pivotal in interpreting complex imaging data, offering rapid, accurate diagnoses and potentially surpassing traditional diagnostic methods. Studies highlighted within the article demonstrate AI's capabilities in accurately detecting conditions such as fractures, pneumothorax, and pulmonary diseases from various imaging modalities including X-rays, CT scans, and MRIs. Furthermore, AI's ability to predict clinical outcomes like mechanical ventilation needs illustrates its potential in crisis resource optimization. Despite these advancements, the integration of AI into clinical practice presents challenges such as data privacy, algorithmic bias, and the need for extensive validation across diverse settings. This review underscores the transformative potential of AI in emergency settings, advocating for a future where AI and clinical expertise synergize to elevate patient care standards. 

**Abstract (ZH)**: 人工智能（AI）正通过增强诊断流程和提升患者结果来革新急诊医学。本文 review 了过去五年AI在急诊影像学应用中的进展，重点介绍了机器学习和深度学习等AI技术在解读复杂影像数据中的关键作用，以及其在预测临床结果方面的潜力。文章中的研究表明，AI能够从多种影像学检查方法（如X光、CT扫描和MRI）中准确检测骨折、气胸和肺部疾病等状况。此外，AI预测临床结果如机械通气需求的能力展示了其在危机资源优化中的潜在作用。尽管取得了这些进步，AI在临床实践中的整合仍面临数据隐私、算法偏见和跨多样场景验证的挑战。本文强调了AI在急诊环境中的变革潜力，倡导未来AI与临床专长的协同作用以提升患者护理标准。 

---
# Inteligencia Artificial para la conservación y uso sostenible de la biodiversidad, una visión desde Colombia (Artificial Intelligence for conservation and sustainable use of biodiversity, a view from Colombia) 

**Title (ZH)**: 人工智能在保护和可持续利用生物多样性中的应用：以哥伦比亚为例 

**Authors**: Juan Sebastián Cañas, Camila Parra-Guevara, Manuela Montoya-Castrillón, Julieta M Ramírez-Mejía, Gabriel-Alejandro Perilla, Esteban Marentes, Nerieth Leuro, Jose Vladimir Sandoval-Sierra, Sindy Martinez-Callejas, Angélica Díaz, Mario Murcia, Elkin A. Noguera-Urbano, Jose Manuel Ochoa-Quintero, Susana Rodríguez Buriticá, Juan Sebastián Ulloa  

**Link**: [PDF](https://arxiv.org/pdf/2503.14543)  

**Abstract**: The rise of artificial intelligence (AI) and the aggravating biodiversity crisis have resulted in a research area where AI-based computational methods are being developed to act as allies in conservation, and the sustainable use and management of natural resources. While important general guidelines have been established globally regarding the opportunities and challenges that this interdisciplinary research offers, it is essential to generate local reflections from the specific contexts and realities of each region. Hence, this document aims to analyze the scope of this research area from a perspective focused on Colombia and the Neotropics. In this paper, we summarize the main experiences and debates that took place at the Humboldt Institute between 2023 and 2024 in Colombia. To illustrate the variety of promising opportunities, we present current uses such as automatic species identification from images and recordings, species modeling, and in silico bioprospecting, among others. From the experiences described above, we highlight limitations, challenges, and opportunities for in order to successfully implementate AI in conservation efforts and sustainable management of biological resources in the Neotropics. The result aims to be a guide for researchers, decision makers, and biodiversity managers, facilitating the understanding of how artificial intelligence can be effectively integrated into conservation and sustainable use strategies. Furthermore, it also seeks to open a space for dialogue on the development of policies that promote the responsible and ethical adoption of AI in local contexts, ensuring that its benefits are harnessed without compromising biodiversity or the cultural and ecosystemic values inherent in Colombia and the Neotropics. 

**Abstract (ZH)**: 人工智能兴起与生物多样性危机加剧背景下基于人工智能的计算方法在哥伦比亚和新热带地区的保护及自然资源可持续利用中的作用研究 

---
# SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders 

**Title (ZH)**: SAUCE: 基于稀疏自编码器的选择性概念遗忘在视觉-语言模型中的应用 

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2503.14530)  

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs. 

**Abstract (ZH)**: 基于稀疏自编码器的视觉-语言模型细粒度选择性去学习方法 

---
# Accessibility Considerations in the Development of an AI Action Plan 

**Title (ZH)**: AI行动计划中的可达性考虑 

**Authors**: Jennifer Mankoff, Janice Light, James Coughlan, Christian Vogler, Abraham Glasser, Gregg Vanderheiden, Laura Rice  

**Link**: [PDF](https://arxiv.org/pdf/2503.14522)  

**Abstract**: We argue that there is a need for Accessibility to be represented in several important domains:
- Capitalize on the new capabilities AI provides - Support for open source development of AI, which can allow disabled and disability focused professionals to contribute, including
- Development of Accessibility Apps which help realise the promise of AI in accessibility domains
- Open Source Model Development and Validation to ensure that accessibility concerns are addressed in these algorithms
- Data Augmentation to include accessibility in data sets used to train models
- Accessible Interfaces that allow disabled people to use any AI app, and to validate its outputs
- Dedicated Functionality and Libraries that can make it easy to integrate AI support into a variety of settings and apps. - Data security and privacy and privacy risks including data collected by AI based accessibility technologies; and the possibility of disability disclosure. - Disability-specific AI risks and biases including both direct bias (during AI use by the disabled person) and indirect bias (when AI is used by someone else on data relating to a disabled person). 

**Abstract (ZH)**: 我们argue需要在以下几个重要领域体现Accessibility：
- 充分利用AI提供的新能力 - 支持开放源代码的AI开发，这可以让残疾和以残疾人为中心的专业人员贡献自己的力量
- 发展无障碍应用程序，以实现AI在无障碍领域的潜力
- 开放源代码模型开发和验证，确保在这些算法中解决无障碍问题
- 数据增强，将无障碍纳入用于训练模型的数据集中
- 可访问界面，使残疾人能够使用任何AI应用程序，并验证其输出
- 专门的功能和库，使其更容易将AI支持集成到各种环境和应用程序中
- 数据安全与隐私以及隐私风险，包括AI无障碍技术收集的数据；以及披露残疾人身份的可能性
- 专门的AI风险和偏见，包括直接偏见（残疾人使用AI期间）和间接偏见（他人使用AI处理与残疾人相关的数据时）。 

---
# Content ARCs: Decentralized Content Rights in the Age of Generative AI 

**Title (ZH)**: 生成人工智能时代的内容权利ARC：去中心化的内容权利管理 

**Authors**: Kar Balan, Andrew Gilbert, John Collomosse  

**Link**: [PDF](https://arxiv.org/pdf/2503.14519)  

**Abstract**: The rise of Generative AI (GenAI) has sparked significant debate over balancing the interests of creative rightsholders and AI developers. As GenAI models are trained on vast datasets that often include copyrighted material, questions around fair compensation and proper attribution have become increasingly urgent. To address these challenges, this paper proposes a framework called \emph{Content ARCs} (Authenticity, Rights, Compensation). By combining open standards for provenance and dynamic licensing with data attribution, and decentralized technologies, Content ARCs create a mechanism for managing rights and compensating creators for using their work in AI training. We characterize several nascent works in the AI data licensing space within Content ARCs and identify where challenges remain to fully implement the end-to-end framework. 

**Abstract (ZH)**: Generative AI (GenAI)的发展引发了关于创造性权利持有者利益与AI开发者之间平衡的显著 debate。由于GenAI模型在训练过程中通常包含了受版权保护的内容，因此关于公平补偿和适当归属的问题日益迫切。为应对这些挑战，本文提出了一种名为Content ARCs（Authenticity, Rights, Compensation）的框架。通过结合溯源的开放标准、动态许可与数据归属，并利用去中心化技术，Content ARCs 创建了一个管理权利和补偿创作者在AI训练中使用其作品的机制。我们对AI数据许可领域的几种初步工作在Content ARCs中的特征进行描述，并识别出实施端到端框架仍存在的挑战。 

---
# Acceptance or Rejection of Lots while Minimizing and Controlling Type I and Type II Errors 

**Title (ZH)**: 接受或拒绝批次并同时最小化和控制类型I和类型II错误 

**Authors**: Edson Luiz Ursini, Elaine Cristina Catapani Poletti, Loreno Menezes da Silveira, José Roberto Emiliano Leite  

**Link**: [PDF](https://arxiv.org/pdf/2503.14514)  

**Abstract**: The double hypothesis test (DHT) is a test that allows controlling Type I (producer) and Type II (consumer) errors. It is possible to say whether the batch has a defect rate, p, between 1.5 and 2%, or between 2 and 5%, or between 5 and 10%, and so on, until finding a required value for this probability. Using the two probabilities side by side, the Type I error for the lower probability distribution and the Type II error for the higher probability distribution, both can be controlled and minimized. It can be applied in the development or manufacturing process of a batch of components, or in the case of purchasing from a supplier, when the percentage of defects (p) is unknown, considering the technology and/or process available to obtain them. The power of the test is amplified by the joint application of the Limit of Successive Failures (LSF) related to the Renewal Theory. To enable the choice of the most appropriate algorithm for each application. Four distributions are proposed for the Bernoulli event sequence, including their computational efforts: Binomial, Binomial approximated by Poisson, and Binomial approximated by Gaussian (with two variants). Fuzzy logic rules are also applied to facilitate decision-making. 

**Abstract (ZH)**: 双假设检验(DHT)是一种能够控制生产者(I型)错误和消费者(II型)错误的检验方法。可以通过计算既可以判断批产品的缺陷率p处于1.5%-2%、2%-5%、5%-10%等区间中的哪一个，直到找到所需概率的值。将两个概率值并列使用，可以同时控制和最小化较低概率分布的I型错误和较高概率分布的II型错误。该方法可以应用于组件批次的研发或制造过程，或在从供应商采购时，当未知缺陷百分比(p)的情况下，结合可用的技术和/或工艺来获取它们。通过与更新理论相关的连续失败上限(LSF)的联合应用，提高了检验的功效。为了适应不同应用选择最合适的算法。提出了四种伯努利事件序列的分布，并讨论了各自的计算成本：二项式分布、泊松逼近的二项式分布、高斯逼近的二项式分布（两种变体），还应用了模糊逻辑规则以辅助决策。 

---
