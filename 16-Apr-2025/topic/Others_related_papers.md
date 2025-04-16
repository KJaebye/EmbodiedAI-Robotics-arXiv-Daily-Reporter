# Neural Control Barrier Functions from Physics Informed Neural Networks 

**Title (ZH)**: 基于物理知情神经网络的神经控制屏障函数 

**Authors**: Shreenabh Agrawal, Manan Tayal, Aditya Singh, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2504.11045)  

**Abstract**: As autonomous systems become increasingly prevalent in daily life, ensuring their safety is paramount. Control Barrier Functions (CBFs) have emerged as an effective tool for guaranteeing safety; however, manually designing them for specific applications remains a significant challenge. With the advent of deep learning techniques, recent research has explored synthesizing CBFs using neural networks-commonly referred to as neural CBFs. This paper introduces a novel class of neural CBFs that leverages a physics-inspired neural network framework by incorporating Zubov's Partial Differential Equation (PDE) within the context of safety. This approach provides a scalable methodology for synthesizing neural CBFs applicable to high-dimensional systems. Furthermore, by utilizing reciprocal CBFs instead of zeroing CBFs, the proposed framework allows for the specification of flexible, user-defined safe regions. To validate the effectiveness of the approach, we present case studies on three different systems: an inverted pendulum, autonomous ground navigation, and aerial navigation in obstacle-laden environments. 

**Abstract (ZH)**: 随着自主系统在日常生活中日益普及，确保其安全变得至关重要。控制障碍函数（CBFs）已成为确保安全的有效工具；然而，为特定应用手工设计它们仍然是一个重大挑战。随着深度学习技术的发展，最近的研究探索了使用神经网络合成CBFs——通常称为神经CBFs。本文介绍了一种新的神经CBF类，通过在安全性框架中结合Zubov偏微分方程（PDE）来利用物理启发的神经网络框架。该方法提供了一种可扩展的合成神经CBFs的方法，适用于高维系统。此外，通过使用逆CBFs而不是零CBFs，所提出的框架允许为用户定义灵活的安全区域。为了验证该方法的有效性，我们在三个不同的系统上进行了案例研究：倒立摆、自主地面导航和障碍环境下的空中导航。 

---
# Acquisition of high-quality images for camera calibration in robotics applications via speech prompts 

**Title (ZH)**: 通过语音提示获取用于机器人应用-camera标定的高质量图像 

**Authors**: Timm Linder, Kadir Yilmaz, David B. Adrian, Bastian Leibe  

**Link**: [PDF](https://arxiv.org/pdf/2504.11031)  

**Abstract**: Accurate intrinsic and extrinsic camera calibration can be an important prerequisite for robotic applications that rely on vision as input. While there is ongoing research on enabling camera calibration using natural images, many systems in practice still rely on using designated calibration targets with e.g. checkerboard patterns or April tag grids. Once calibration images from different perspectives have been acquired and feature descriptors detected, those are typically used in an optimization process to minimize the geometric reprojection error. For this optimization to converge, input images need to be of sufficient quality and particularly sharpness; they should neither contain motion blur nor rolling-shutter artifacts that can arise when the calibration board was not static during image capture. In this work, we present a novel calibration image acquisition technique controlled via voice commands recorded with a clip-on microphone, that can be more robust and user-friendly than e.g. triggering capture with a remote control, or filtering out blurry frames from a video sequence in postprocessing. To achieve this, we use a state-of-the-art speech-to-text transcription model with accurate per-word timestamping to capture trigger words with precise temporal alignment. Our experiments show that the proposed method improves user experience by being fast and efficient, allowing us to successfully calibrate complex multi-camera setups. 

**Abstract (ZH)**: 准确的固有和外在相机标定对于依赖视觉输入的机器人应用来说是重要的先决条件。尽管目前有关利用自然图像进行相机标定的研究正在进行，但在实践中许多系统仍然依靠使用特定的标定目标，如棋盘格模式或AprilTag网格。一旦从不同视角获取了标定图像并检测出特征描述符，通常会将这些特征描述符用于优化过程以最小化几何重投影误差。为了使这种优化能够收敛，输入图像需要具有足够的质量，特别是在捕获图像时标定板应该是静止的，以避免运动模糊或滚筒快门伪影。在本工作中，我们提出了一种通过贴片麦克风录制语音命令控制的新型标定图像采集技术，该技术比使用遥控器触发捕获或在后期处理中过滤模糊帧更加健壮和用户友好。为实现这一目标，我们使用了具有准确逐词时间戳的最新演讲转文字模型，以实现精确的时间对齐。我们的实验表明，所提出的方法通过提高高效性改善了用户体验，使我们能够成功标定复杂的多相机系统。 

---
# A Sublinear Algorithm for Path Feasibility Among Rectangular Obstacles 

**Title (ZH)**: 亚线性算法实现矩形障碍物路径可行性的判断 

**Authors**: Alex Fan, Alicia Li, Arul Kolla, Jason Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2504.10859)  

**Abstract**: The problem of finding a path between two points while avoiding obstacles is critical in robotic path planning. We focus on the feasibility problem: determining whether such a path exists. We model the robot as a query-specific rectangular object capable of moving parallel to its sides. The obstacles are axis-aligned, rectangular, and may overlap. Most previous works only consider nondisjoint rectangular objects and point-sized or statically sized robots. Our approach introduces a novel technique leveraging generalized Gabriel graphs and constructs a data structure to facilitate online queries regarding path feasibility with varying robot sizes in sublinear time. To efficiently handle feasibility queries, we propose an online algorithm utilizing sweep line to construct a generalized Gabriel graph under the $L_\infty$ norm, capturing key gap constraints between obstacles. We utilize a persistent disjoint-set union data structure to efficiently determine feasibility queries in $\mathcal{O}(\log n)$ time and $\mathcal{O}(n)$ total space. 

**Abstract (ZH)**: 在两点之间寻找避免障碍物的路径问题是机器人路径规划中的关键问题。我们关注可行性问题：确定这样的路径是否存在。我们将机器人建模为查询特定的矩形对象，能够在其边上平行移动。障碍物是轴对齐的矩形，并且可能重叠。大多数以前的工作只考虑非分离的矩形对象和点大小或静态大小的机器人。我们的方法引入了一种新的技术，利用广义加布里埃尔图，并构建了一个数据结构，以实现在线查询路径可行性，同时随着机器人大小的变化在亚线性时间内进行。为了高效地处理可行性查询，我们提出了一种利用扫描线在线构建在$L_\infty$范数下的广义加布里埃尔图的算法，捕捉障碍物之间的关键间隙约束。我们利用持久分离集合合并数据结构，在$\mathcal{O}(\log n)$时间复杂度和$\mathcal{O}(n)$总空间复杂度下高效地确定可行性查询。 

---
# CleanMAP: Distilling Multimodal LLMs for Confidence-Driven Crowdsourced HD Map Updates 

**Title (ZH)**: CleanMAP：提炼多模态LLMs以实现基于信心驱动的高精度地图众包更新 

**Authors**: Ankit Kumar Shaw, Kun Jiang, Tuopu Wen, Chandan Kumar Sah, Yining Shi, Mengmeng Yang, Diange Yang, Xiaoli Lian  

**Link**: [PDF](https://arxiv.org/pdf/2504.10738)  

**Abstract**: The rapid growth of intelligent connected vehicles (ICVs) and integrated vehicle-road-cloud systems has increased the demand for accurate, real-time HD map updates. However, ensuring map reliability remains challenging due to inconsistencies in crowdsourced data, which suffer from motion blur, lighting variations, adverse weather, and lane marking degradation. This paper introduces CleanMAP, a Multimodal Large Language Model (MLLM)-based distillation framework designed to filter and refine crowdsourced data for high-confidence HD map updates. CleanMAP leverages an MLLM-driven lane visibility scoring model that systematically quantifies key visual parameters, assigning confidence scores (0-10) based on their impact on lane detection. A novel dynamic piecewise confidence-scoring function adapts scores based on lane visibility, ensuring strong alignment with human evaluations while effectively filtering unreliable data. To further optimize map accuracy, a confidence-driven local map fusion strategy ranks and selects the top-k highest-scoring local maps within an optimal confidence range (best score minus 10%), striking a balance between data quality and quantity. Experimental evaluations on a real-world autonomous vehicle dataset validate CleanMAP's effectiveness, demonstrating that fusing the top three local maps achieves the lowest mean map update error of 0.28m, outperforming the baseline (0.37m) and meeting stringent accuracy thresholds (<= 0.32m). Further validation with real-vehicle data confirms 84.88% alignment with human evaluators, reinforcing the model's robustness and reliability. This work establishes CleanMAP as a scalable and deployable solution for crowdsourced HD map updates, ensuring more precise and reliable autonomous navigation. The code will be available at this https URL 

**Abstract (ZH)**: 基于多模态大型语言模型的CleanMAP地图清洁框架：确保高精度的 crowdsourced 高精地图更新 

---
# C-SHAP for time series: An approach to high-level temporal explanations 

**Title (ZH)**: C-SHAP for 时间序列：一种高层次时间解释的方法 

**Authors**: Annemarie Jutte, Faizan Ahmed, Jeroen Linssen, Maurice van Keulen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11159)  

**Abstract**: Time series are ubiquitous in domains such as energy forecasting, healthcare, and industry. Using AI systems, some tasks within these domains can be efficiently handled. Explainable AI (XAI) aims to increase the reliability of AI solutions by explaining model reasoning. For time series, many XAI methods provide point- or sequence-based attribution maps. These methods explain model reasoning in terms of low-level patterns. However, they do not capture high-level patterns that may also influence model reasoning. We propose a concept-based method to provide explanations in terms of these high-level patterns. In this paper, we present C-SHAP for time series, an approach which determines the contribution of concepts to a model outcome. We provide a general definition of C-SHAP and present an example implementation using time series decomposition. Additionally, we demonstrate the effectiveness of the methodology through a use case from the energy domain. 

**Abstract (ZH)**: 时间序列在能源预测、医疗健康和工业等领域广泛存在。通过AI系统，这些领域的某些任务可以得到有效处理。可解释AI（XAI）旨在通过解释模型推理来提高AI解决方案的可靠性。对于时间序列，许多XAI方法提供基于点或序列的归因图，这些方法从低级模式的角度解释模型推理，但没有捕捉到可能也影响模型推理的高级模式。我们提出了一种基于概念的方法，以这些高级模式来提供解释。在本文中，我们介绍了时间序列的C-SHAP方法，该方法确定概念对模型结果的贡献。我们提供了C-SHAP的通用定义，并通过时间序列分解示例展示了其实现方式。此外，我们通过能源领域的案例研究展示了该方法的有效性。 

---
# Understanding the theoretical properties of projected Bellman equation, linear Q-learning, and approximate value iteration 

**Title (ZH)**: 理解投影贝尔曼方程、线性Q学习和近似值迭代的理论性质 

**Authors**: Han-Dong Lim, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.10865)  

**Abstract**: In this paper, we study the theoretical properties of the projected Bellman equation (PBE) and two algorithms to solve this equation: linear Q-learning and approximate value iteration (AVI). We consider two sufficient conditions for the existence of a solution to PBE : strictly negatively row dominating diagonal (SNRDD) assumption and a condition motivated by the convergence of AVI. The SNRDD assumption also ensures the convergence of linear Q-learning, and its relationship with the convergence of AVI is examined. Lastly, several interesting observations on the solution of PBE are provided when using $\epsilon$-greedy policy. 

**Abstract (ZH)**: 在本文中，我们研究了投影贝尔曼方程（PBE）的理论性质以及解决该方程的两种算法：线性Q学习和近似值迭代（AVI）。我们考虑了PBE解存在的两个充分条件：严格负行支配对角（SNRDD）假设以及一种受AVI收敛性启发的条件。SNRDD假设还保证了线性Q学习的收敛性，并探讨了其与AVI收敛性之间的关系。最后，我们提供了在使用$\epsilon$-贪婪策略时PBE解的一些有趣观察。 

---
# Ride-pool Assignment Algorithms: Modern Implementation and Swapping Heuristics 

**Title (ZH)**: 拼车分配算法：现代实现与置换 heuristic 研究 

**Authors**: Matthew Zalesak, Hins Hu, Samitha Samaranayake  

**Link**: [PDF](https://arxiv.org/pdf/2504.10649)  

**Abstract**: On-demand ride-pooling has emerged as a popular urban transportation solution, addressing the efficiency limitations of traditional ride-hailing services by grouping multiple riding requests with spatiotemporal proximity into a single vehicle. Although numerous algorithms have been developed for the Ride-pool Assignment Problem (RAP) -- a core component of ride-pooling systems, there is a lack of open-source implementations, making it difficult to benchmark these algorithms on a common dataset and objective. In this paper, we present the implementation details of a ride-pool simulator that encompasses several key ride-pool assignment algorithms, along with associated components such as vehicle routing and rebalancing. We also open-source a highly optimized and modular C++ codebase, designed to facilitate the extension of new algorithms and features. Additionally, we introduce a family of swapping-based local-search heuristics to enhance existing ride-pool assignment algorithms, achieving a better balance between performance and computational efficiency. Extensive experiments on a large-scale, real-world dataset from Manhattan, NYC reveal that while all selected algorithms perform comparably, the newly proposed Multi-Round Linear Assignment with Cyclic Exchange (LA-MR-CE) algorithm achieves a state-of-the-art service rate with significantly reduced computational time. Furthermore, an in-depth analysis suggests that a performance barrier exists for all myopic ride-pool assignment algorithms due to the system's capacity bottleneck, and incorporating future information could be key to overcoming this limitation. 

**Abstract (ZH)**: 按需拼车已成为一种流行的都市交通解决方案，通过将具有时空临近性的多个乘车请求分组到同一辆车中，解决了传统拼车服务的效率限制。尽管已开发出多种用于乘车拼组分配问题（RAP）的算法——这是拼车系统的核心组成部分之一，但缺乏开源实现，使得在共同的数据集和目标上对这些算法进行基准测试变得困难。本文介绍了包含多种关键乘车拼组分配算法及相关组件（如车辆路由和再平衡）的乘车拼组模拟器的实现细节。我们还开源了一个高度优化且模块化的C++代码库，旨在方便新算法和功能的扩展。此外，我们引入了一类基于交换的局部搜索启发式算法，以增强现有的乘车拼组分配算法，实现性能和计算效率之间的更好平衡。大规模现实世界数据集（来自纽约市曼哈顿区）的实验结果显示，虽然所有选定的算法表现相当，但新提出的多轮线性分配伴有循环交换（LA-MR-CE）算法在显著减少计算时间的同时实现了最先进的服务率。进一步的分析表明，由于系统容量瓶颈，所有短视的乘车拼组分配算法都存在性能障碍，而引入未来信息可能是克服这一限制的关键。 

---
# Explainable Artificial Intelligence techniques for interpretation of food datasets: a review 

**Title (ZH)**: 可解释的人工智能技术在食品数据集解释中的应用：一个综述 

**Authors**: Leonardo Arrighi, Ingrid Alves de Moraes, Marco Zullich, Michele Simonato, Douglas Fernandes Barbin, Sylvio Barbon Junior  

**Link**: [PDF](https://arxiv.org/pdf/2504.10527)  

**Abstract**: Artificial Intelligence (AI) has become essential for analyzing complex data and solving highly-challenging tasks. It is being applied across numerous disciplines beyond computer science, including Food Engineering, where there is a growing demand for accurate and trustworthy predictions to meet stringent food quality standards. However, this requires increasingly complex AI models, raising reliability concerns. In response, eXplainable AI (XAI) has emerged to provide insights into AI decision-making, aiding model interpretation by developers and users. Nevertheless, XAI remains underutilized in Food Engineering, limiting model reliability. For instance, in food quality control, AI models using spectral imaging can detect contaminants or assess freshness levels, but their opaque decision-making process hinders adoption. XAI techniques such as SHAP (Shapley Additive Explanations) and Grad-CAM (Gradient-weighted Class Activation Mapping) can pinpoint which spectral wavelengths or image regions contribute most to a prediction, enhancing transparency and aiding quality control inspectors in verifying AI-generated assessments. This survey presents a taxonomy for classifying food quality research using XAI techniques, organized by data types and explanation methods, to guide researchers in choosing suitable approaches. We also highlight trends, challenges, and opportunities to encourage the adoption of XAI in Food Engineering. 

**Abstract (ZH)**: 人工智能（AI）已成为分析复杂数据和解决高度挑战性任务的重要工具。它已跨计算机科学以外的多个学科得到应用，包括食品工程，该领域对准确和可信赖的预测需求越来越大，以满足严格的食物质量标准。然而，这要求使用越来越复杂的AI模型，从而引发可靠性方面的担忧。为此，可解释人工智能（XAI）已经出现，以提供对AI决策的洞察，帮助开发者和用户理解和解释模型。尽管如此，XAI在食品工程中的应用仍然不足，限制了模型的可靠性。例如，在食品质量控制中，使用光谱成像的AI模型可以检测污染物或评估新鲜度水平，但它们不透明的决策过程阻碍了其应用。可解释AI技术，如SHAP（Shapley值分解）和Grad-CAM（梯度加权类激活映射），可以识别出对预测贡献最大的光谱波长或图像区域，提高透明度，并帮助质量控制检查员验证AI生成的评估结果。本文综述了使用XAI技术分类食品质量研究的分类体系，按数据类型和解释方法组织，以指导研究人员选择合适的方法。我们还指出了趋势、挑战和机遇，以促进XAI在食品工程中的应用。 

---
# Elucidating the Design Space of Multimodal Protein Language Models 

**Title (ZH)**: 阐述多模态蛋白质语言模型的设计空间 

**Authors**: Cheng-Yen, Hsieh, Xinyou Wang, Daiheng Zhang, Dongyu Xue, Fei Ye, Shujian Huang, Zaixiang Zheng, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11454)  

**Abstract**: Multimodal protein language models (PLMs) integrate sequence and token-based structural information, serving as a powerful foundation for protein modeling, generation, and design. However, the reliance on tokenizing 3D structures into discrete tokens causes substantial loss of fidelity about fine-grained structural details and correlations. In this paper, we systematically elucidate the design space of multimodal PLMs to overcome their limitations. We identify tokenization loss and inaccurate structure token predictions by the PLMs as major bottlenecks. To address these, our proposed design space covers improved generative modeling, structure-aware architectures and representation learning, and data exploration. Our advancements approach finer-grained supervision, demonstrating that token-based multimodal PLMs can achieve robust structural modeling. The effective design methods dramatically improve the structure generation diversity, and notably, folding abilities of our 650M model by reducing the RMSD from 5.52 to 2.36 on PDB testset, even outperforming 3B baselines and on par with the specialized folding models. 

**Abstract (ZH)**: 多模态蛋白质语言模型的设计空间：克服限制实现精细结构建模 

---
# Greedy Restart Schedules: A Baseline for Dynamic Algorithm Selection on Numerical Black-box Optimization Problems 

**Title (ZH)**: 贪婪重启调度：数值黑箱优化问题中动态算法选择的基线方法 

**Authors**: Lennart Schäpermeier  

**Link**: [PDF](https://arxiv.org/pdf/2504.11440)  

**Abstract**: In many optimization domains, there are multiple different solvers that contribute to the overall state-of-the-art, each performing better on some, and worse on other types of problem instances. Meta-algorithmic approaches, such as instance-based algorithm selection, configuration and scheduling, aim to close this gap by extracting the most performance possible from a set of (configurable) optimizers. In this context, the best performing individual algorithms are often hand-crafted hybrid heuristics which perform many restarts of fast local optimization approaches. However, data-driven techniques to create optimized restart schedules have not yet been extensively studied.
Here, we present a simple scheduling approach that iteratively selects the algorithm performing best on the distribution of unsolved training problems at time of selection, resulting in a problem-independent solver schedule. We demonstrate our approach using well-known optimizers from numerical black-box optimization on the BBOB testbed, bridging much of the gap between single and virtual best solver from the original portfolio across various evaluation protocols. Our greedy restart schedule presents a powerful baseline for more complex dynamic algorithm selection models. 

**Abstract (ZH)**: 在多个优化领域中，存在多种不同的求解器，各自在不同类型的优化问题上表现出色或较差。元算法方法，如基于实例的算法选择、配置和调度，旨在通过从一组（可配置的）优化器中提取最佳性能来缩小这一差距。在此背景下，表现最好的单独算法通常是手工构建的混合启发式算法，这些算法在快速局部优化方法的基础上进行多次重启。然而，利用数据驱动技术创建优化重启计划的时间表尚未被广泛研究。在这里，我们提出了一种简单的时间表方法，该方法在选择时间点上迭代选择在未解决的训练问题分布上表现最佳的算法，从而生成一个与问题无关的求解器时间序列。我们使用数值黑盒优化中的知名优化器在BBOB测试台上展示了该方法，该方法在各种评估协议下极大地缩小了单一最优求解器与虚拟最优求解器之间的时间隔。我们的贪婪重启计划为更复杂的动态算法选择模型提供了强大的基线。 

---
# Measures of Variability for Risk-averse Policy Gradient 

**Title (ZH)**: 风险回避策略梯度的变异性度量 

**Authors**: Yudong Luo, Yangchen Pan, Jiaqi Tan, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2504.11412)  

**Abstract**: Risk-averse reinforcement learning (RARL) is critical for decision-making under uncertainty, which is especially valuable in high-stake applications. However, most existing works focus on risk measures, e.g., conditional value-at-risk (CVaR), while measures of variability remain underexplored. In this paper, we comprehensively study nine common measures of variability, namely Variance, Gini Deviation, Mean Deviation, Mean-Median Deviation, Standard Deviation, Inter-Quantile Range, CVaR Deviation, Semi_Variance, and Semi_Standard Deviation. Among them, four metrics have not been previously studied in RARL. We derive policy gradient formulas for these unstudied metrics, improve gradient estimation for Gini Deviation, analyze their gradient properties, and incorporate them with the REINFORCE and PPO frameworks to penalize the dispersion of returns.
Our empirical study reveals that variance-based metrics lead to unstable policy updates. In contrast, CVaR Deviation and Gini Deviation show consistent performance across different randomness and evaluation domains, achieving high returns while effectively learning risk-averse policies. Mean Deviation and Semi_Standard Deviation are also competitive across different scenarios. This work provides a comprehensive overview of variability measures in RARL, offering practical insights for risk-aware decision-making and guiding future research on risk metrics and RARL algorithms. 

**Abstract (ZH)**: 风险规避强化学习 (RARL) 在不确定性决策中至关重要，特别是在高风险应用场景中尤为有价值。然而，现有大部分工作集中在风险度量上，如条件值-at-风险 (CVaR)，而波动性度量则尚未充分探索。在本文中，我们全面研究了九种常见的波动性度量，包括方差、基尼偏差、均值偏差、均值-中位数偏差、标准差、分位数间距、CVaR偏差、半方差和半标准差。其中，有四种指标在RARL中尚未被研究。我们推导了这些未研究指标的策略梯度公式，改进了基尼偏差的梯度估计，分析了它们的梯度性质，并将其与REINFORCE和PPO框架结合，用于惩罚回报的分散性。 

---
# Multi-level Cellular Automata for FLIM networks 

**Title (ZH)**: 多级细胞自动机for FLIM网络 

**Authors**: Felipe Crispim Salvagnini, Jancarlo F. Gomes, Cid A. N. Santos, Silvio Jamil F. Guimarães, Alexandre X. Falcão  

**Link**: [PDF](https://arxiv.org/pdf/2504.11406)  

**Abstract**: The necessity of abundant annotated data and complex network architectures presents a significant challenge in deep-learning Salient Object Detection (deep SOD) and across the broader deep-learning landscape. This challenge is particularly acute in medical applications in developing countries with limited computational resources. Combining modern and classical techniques offers a path to maintaining competitive performance while enabling practical applications. Feature Learning from Image Markers (FLIM) methodology empowers experts to design convolutional encoders through user-drawn markers, with filters learned directly from these annotations. Recent findings demonstrate that coupling a FLIM encoder with an adaptive decoder creates a flyweight network suitable for SOD, requiring significantly fewer parameters than lightweight models and eliminating the need for backpropagation. Cellular Automata (CA) methods have proven successful in data-scarce scenarios but require proper initialization -- typically through user input, priors, or randomness. We propose a practical intersection of these approaches: using FLIM networks to initialize CA states with expert knowledge without requiring user interaction for each image. By decoding features from each level of a FLIM network, we can initialize multiple CAs simultaneously, creating a multi-level framework. Our method leverages the hierarchical knowledge encoded across different network layers, merging multiple saliency maps into a high-quality final output that functions as a CA ensemble. Benchmarks across two challenging medical datasets demonstrate the competitiveness of our multi-level CA approach compared to established models in the deep SOD literature. 

**Abstract (ZH)**: 丰富的标注数据和复杂的网络架构在深度学习显著目标检测（deep SOD）及更广泛的深度学习领域中提出了重大挑战。这一挑战在计算资源有限的 developing countries 的医疗应用中尤为严峻。结合现代和经典技术为维持竞争力同时实现实际应用提供了一条途径。图像标记特征学习（FLIM）方法使专家能够通过用户绘制的标记设计卷积编码器，并直接从这些注释中学习滤波器。最近的研究表明，将FLIM编码器与自适应解码器结合使用，可创建一种轻量级网络，所需的参数显著少于轻量级模型，并且不需要反向传播。在数据稀缺情况下，细胞自动机（CA）方法已经证明是有效的，但需要适当的初始化通常通过用户输入、先验知识或随机性实现。我们提出了一种实用的结合方法：利用FLIM网络在不需每张图像都进行用户交互的情况下，通过专家知识初始化CA状态。通过从FLIM网络的每一层解码特征，我们可以同时初始化多个CA，创建一个多层框架。该方法利用了不同网络层中编码的层次知识，将多个显著性图合并成高质量的最终输出，作为CA的集成。在两个具有挑战性的医学数据集上的基准测试表明，与深度SOD文献中已有的模型相比，我们的多层CA方法具有竞争力。 

---
# Trajectory Encoding Temporal Graph Networks 

**Title (ZH)**: 轨迹编码时序图网络 

**Authors**: Jiafeng Xiong, Rizos Sakellariou  

**Link**: [PDF](https://arxiv.org/pdf/2504.11386)  

**Abstract**: Temporal Graph Networks (TGNs) have demonstrated significant success in dynamic graph tasks such as link prediction and node classification. Both tasks comprise transductive settings, where the model predicts links among known nodes, and in inductive settings, where it generalises learned patterns to previously unseen nodes. Existing TGN designs face a dilemma under these dual scenarios. Anonymous TGNs, which rely solely on temporal and structural information, offer strong inductive generalisation but struggle to distinguish known nodes. In contrast, non-anonymous TGNs leverage node features to excel in transductive tasks yet fail to adapt to new nodes. To address this challenge, we propose Trajectory Encoding TGN (TETGN). Our approach introduces automatically expandable node identifiers (IDs) as learnable temporal positional features and performs message passing over these IDs to capture each node's historical context. By integrating this trajectory-aware module with a standard TGN using multi-head attention, TETGN effectively balances transductive accuracy with inductive generalisation. Experimental results on three real-world datasets show that TETGN significantly outperforms strong baselines on both link prediction and node classification tasks, demonstrating its ability to unify the advantages of anonymous and non-anonymous models for dynamic graph learning. 

**Abstract (ZH)**: Temporal Graph Networks with Trajectory-aware Encoding for Dynamic Graph Learning 

---
# Neural Networks for on-chip Model Predictive Control: a Method to Build Optimized Training Datasets and its application to Type-1 Diabetes 

**Title (ZH)**: 基于神经网络的片上模型预测控制方法及其在1型糖尿病中的应用：构建优化训练数据集的方法 

**Authors**: Alberto Castillo, Elliot Pryor, Anas El Fathi, Boris Kovatchev, Marc Breton  

**Link**: [PDF](https://arxiv.org/pdf/2504.11355)  

**Abstract**: Training Neural Networks (NNs) to behave as Model Predictive Control (MPC) algorithms is an effective way to implement them in constrained embedded devices. By collecting large amounts of input-output data, where inputs represent system states and outputs are MPC-generated control actions, NNs can be trained to replicate MPC behavior at a fraction of the computational cost. However, although the composition of the training data critically influences the final NN accuracy, methods for systematically optimizing it remain underexplored. In this paper, we introduce the concept of Optimally-Sampled Datasets (OSDs) as ideal training sets and present an efficient algorithm for generating them. An OSD is a parametrized subset of all the available data that (i) preserves existing MPC information up to a certain numerical resolution, (ii) avoids duplicate or near-duplicate states, and (iii) becomes saturated or complete. We demonstrate the effectiveness of OSDs by training NNs to replicate the University of Virginia's MPC algorithm for automated insulin delivery in Type-1 Diabetes, achieving a four-fold improvement in final accuracy. Notably, two OSD-trained NNs received regulatory clearance for clinical testing as the first NN-based control algorithm for direct human insulin dosing. This methodology opens new pathways for implementing advanced optimizations on resource-constrained embedded platforms, potentially revolutionizing how complex algorithms are deployed. 

**Abstract (ZH)**: 将神经网络训练为模型预测控制算法可以有效地在约束嵌入式设备中实现它们。通过收集大量输入-输出数据，其中输入代表系统状态，输出是模型预测控制生成的控制动作，神经网络可以以极低的计算成本复制模型预测控制的行为。然而，尽管训练数据的组成对最终神经网络的准确性至关重要，但系统地优化它的方法仍然研究不足。在本文中，我们提出了最优采样数据集（OSD）的概念，作为理想的训练集，并提出了一种高效生成它们的算法。OSD是一个参数化的可用数据子集，它(i)保留到一定数值分辨率的现有模型预测控制信息，(ii)避免重复或近乎重复的状态，(iii)变得饱和或完备。我们通过训练神经网络复制弗吉尼亚大学的模型预测控制算法来自动输送胰岛素，在1型糖尿病中实现了四倍的最终准确性提升。值得注意的是，两种经过OSD训练的神经网络获得了生物医学监管机构的临床测试批准，成为第一个基于神经网络的直接人类胰岛素剂量控制算法。这种方法为资源受限的嵌入式平台上的高级优化提供了新的途径，有可能彻底改变复杂算法的部署方式。 

---
# Interpretable Hybrid-Rule Temporal Point Processes 

**Title (ZH)**: 可解释的混合规则时序点过程 

**Authors**: Yunyang Cao, Juekai Lin, Hongye Wang, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.11344)  

**Abstract**: Temporal Point Processes (TPPs) are widely used for modeling event sequences in various medical domains, such as disease onset prediction, progression analysis, and clinical decision support. Although TPPs effectively capture temporal dynamics, their lack of interpretability remains a critical challenge. Recent advancements have introduced interpretable TPPs. However, these methods fail to incorporate numerical features, thereby limiting their ability to generate precise predictions. To address this issue, we propose Hybrid-Rule Temporal Point Processes (HRTPP), a novel framework that integrates temporal logic rules with numerical features, improving both interpretability and predictive accuracy in event modeling. HRTPP comprises three key components: basic intensity for intrinsic event likelihood, rule-based intensity for structured temporal dependencies, and numerical feature intensity for dynamic probability modulation. To effectively discover valid rules, we introduce a two-phase rule mining strategy with Bayesian optimization. To evaluate our method, we establish a multi-criteria assessment framework, incorporating rule validity, model fitting, and temporal predictive accuracy. Experimental results on real-world medical datasets demonstrate that HRTPP outperforms state-of-the-art interpretable TPPs in terms of predictive performance and clinical interpretability. In case studies, the rules extracted by HRTPP explain the disease progression, offering valuable contributions to medical diagnosis. 

**Abstract (ZH)**: 混合规则时序点过程：兼具解释性和预测精度的新型时序事件建模框架 

---
# Transformer-Based Model for Cold Start Mitigation in FaaS Architecture 

**Title (ZH)**: 基于Transformer的模型在FaaS架构中的冷启动缓解 

**Authors**: Alexandre Savi Fayam Mbala Mouen, Jerry Lacmou Zeutouo, Vianney Kengne Tchendji  

**Link**: [PDF](https://arxiv.org/pdf/2504.11338)  

**Abstract**: Serverless architectures, particularly the Function as a Service (FaaS) model, have become a cornerstone of modern cloud computing due to their ability to simplify resource management and enhance application deployment agility. However, a significant challenge remains: the cold start problem. This phenomenon occurs when an idle FaaS function is invoked, requiring a full initialization process, which increases latency and degrades user experience. Existing solutions for cold start mitigation are limited in terms of invocation pattern generalization and implementation complexity. In this study, we propose an innovative approach leveraging Transformer models to mitigate the impact of cold starts in FaaS architectures. Our solution excels in accurately modeling function initialization delays and optimizing serverless system performance. Experimental evaluation using a public dataset provided by Azure demonstrates a significant reduction in cold start times, reaching up to 79\% compared to conventional methods. 

**Abstract (ZH)**: 无服务器架构，特别是函数即服务（FaaS）模型，因其能够简化资源管理并增强应用部署 agility，已成为现代云计算的基石。然而，一个重大挑战仍然存在：冷启动问题。当一个空闲的FaaS函数被调用时，会触发完整的初始化过程，从而增加延迟并降低用户体验。现有的冷启动缓解解决方案在调用模式泛化和实现复杂性方面受到限制。在本研究中，我们提出了一种创新的方法，利用Transformer模型来缓解FaaS架构中的冷启动影响。我们的解决方案在准确建模函数初始化延迟和优化无服务器系统性能方面表现出色。使用Azure提供的公共数据集进行的实验评估表明，与传统方法相比，冷启动时间显著减少，最高可达79%。 

---
# Looking beyond the next token 

**Title (ZH)**: 超越下一个词 

**Authors**: Abitha Thankaraj, Yiding Jiang, J. Zico Kolter, Yonatan Bisk  

**Link**: [PDF](https://arxiv.org/pdf/2504.11336)  

**Abstract**: The structure of causal language model training assumes that each token can be accurately predicted from the previous context. This contrasts with humans' natural writing and reasoning process, where goals are typically known before the exact argument or phrasings. While this mismatch has been well studied in the literature, the working assumption has been that architectural changes are needed to address this mismatch. We argue that rearranging and processing the training data sequences can allow models to more accurately imitate the true data-generating process, and does not require any other changes to the architecture or training infrastructure. We demonstrate that this technique, Trelawney, and the inference algorithms derived from it allow us to improve performance on several key benchmarks that span planning, algorithmic reasoning, and story generation tasks. Finally, our method naturally enables the generation of long-term goals at no additional cost. We investigate how using the model's goal-generation capability can further improve planning and reasoning. Additionally, we believe Trelawney could potentially open doors to new capabilities beyond the current language modeling paradigm. 

**Abstract (ZH)**: 因果语言模型训练的结构假设每个令牌可以从之前的上下文中准确预测。这与人类自然写作和推理过程不同，在人类过程中，目标通常在具体论据或措辞之前就已经知道了。尽管这种不匹配在文献中已有充分研究，但假定需要对架构进行改变以解决这种不匹配。我们认为重新排列和处理训练数据序列可以让模型更准确地模仿真实的数据生成过程，不需要对架构或训练基础设施进行任何其他修改。我们证明了这种方法Trelawney及其推导出的推理算法能够在多个涵盖规划、算法推理和故事生成的任务基准上改善性能。最后，我们的方法自然地使生成长期目标成为零成本操作。我们研究了利用模型的目标生成能力如何进一步改进规划和推理。此外，我们认为Trelawney可能为超越当前语言模型范式的新型能力打开大门。 

---
# Code Reborn AI-Driven Legacy Systems Modernization from COBOL to Java 

**Title (ZH)**: AI驱动的COBOL到Java的遗产系统现代化重生成代码 

**Authors**: Gopichand Bandarupalli  

**Link**: [PDF](https://arxiv.org/pdf/2504.11335)  

**Abstract**: This study investigates AI-driven modernization of legacy COBOL code into Java, addressing a critical challenge in aging software systems. Leveraging the Legacy COBOL 2024 Corpus -- 50,000 COBOL files from public and enterprise sources -- Java parses the code, AI suggests upgrades, and React visualizes gains. Achieving 93% accuracy, complexity drops 35% (from 18 to 11.7) and coupling 33% (from 8 to 5.4), surpassing manual efforts (75%) and rule-based tools (82%). The approach offers a scalable path to rejuvenate COBOL systems, vital for industries like banking and insurance. 

**Abstract (ZH)**: 本研究探讨了基于AI的老一代COBOL代码向Java的现代化转型，解决了老化软件系统中的关键挑战。利用Legacy COBOL 2024语料库（包含50,000个来自公共和企业来源的COBOL文件），Java解析代码，AI建议更新，并使用React可视化改进效果。实现了93%的准确性，复杂性降低35%（从18降至11.7），耦合度降低33%（从8降至5.4），超过了人工努力（75%）和基于规则的工具（82%）。该方法为银行业和保险业等行业的COBOL系统焕发活力提供了可扩展的路径。 

---
# Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation 

**Title (ZH)**: 多标签下的二部排名：损失函数与标签聚合的研究 

**Authors**: Michal Lukasik, Lin Chen, Harikrishna Narasimhan, Aditya Krishna Menon, Wittawat Jitkrittum, Felix X. Yu, Sashank J. Reddi, Gang Fu, Mohammadhossein Bateni, Sanjiv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.11284)  

**Abstract**: Bipartite ranking is a fundamental supervised learning problem, with the goal of learning a ranking over instances with maximal area under the ROC curve (AUC) against a single binary target label. However, one may often observe multiple binary target labels, e.g., from distinct human annotators. How can one synthesize such labels into a single coherent ranking? In this work, we formally analyze two approaches to this problem -- loss aggregation and label aggregation -- by characterizing their Bayes-optimal solutions. Based on this, we show that while both methods can yield Pareto-optimal solutions, loss aggregation can exhibit label dictatorship: one can inadvertently (and undesirably) favor one label over others. This suggests that label aggregation can be preferable to loss aggregation, which we empirically verify. 

**Abstract (ZH)**: 双部图排序是监督学习中的一个基础问题，目标是在单一二元目标标签下最大化受试操作特征曲线面积（AUC）的实例排名学习。然而，人们常常会观察到多个二元目标标签，例如来自不同的手工标注者。如何将这些标签综合为一个一致的排名？在本文中，我们通过刻画这两种方法的贝叶斯最优解来正式分析合成这些标签为单一一致排名的问题——损失聚合和标签聚合。基于此，我们证明尽管两种方法都可以产生帕累托最优解，但损失聚合可能会表现出标签独裁：可能会无意中（且不希望地）偏好一个标签而忽视其他标签。这表明标签聚合可能比损失聚合更可取，我们在实验中证实了这一点。 

---
# Single-Input Multi-Output Model Merging: Leveraging Foundation Models for Dense Multi-Task Learning 

**Title (ZH)**: 单输入多输出模型融合：利用基础模型进行密集多任务学习 

**Authors**: Juan Garcia Giraldo, Nikolaos Dimitriadis, Ke Wang, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2504.11268)  

**Abstract**: Model merging is a flexible and computationally tractable approach to merge single-task checkpoints into a multi-task model. Prior work has solely focused on constrained multi-task settings where there is a one-to-one mapping between a sample and a task, overlooking the paradigm where multiple tasks may operate on the same sample, e.g., scene understanding. In this paper, we focus on the multi-task setting with single-input-multiple-outputs (SIMO) and show that it qualitatively differs from the single-input-single-output model merging settings studied in the literature due to the existence of task-specific decoders and diverse loss objectives. We identify that existing model merging methods lead to significant performance degradation, primarily due to representation misalignment between the merged encoder and task-specific decoders. We propose two simple and efficient fixes for the SIMO setting to re-align the feature representation after merging. Compared to joint fine-tuning, our approach is computationally effective and flexible, and sheds light into identifying task relationships in an offline manner. Experiments on NYUv2, Cityscapes, and a subset of the Taskonomy dataset demonstrate: (1) task arithmetic suffices to enable multi-task capabilities; however, the representations generated by the merged encoder has to be re-aligned with the task-specific heads; (2) the proposed architecture rivals traditional multi-task learning in performance but requires fewer samples and training steps by leveraging the existence of task-specific models. 

**Abstract (ZH)**: 模型融合是一种灵活且计算上可实现的方法，用于将单任务检查点合并为多任务模型。先前的工作仅专注于具有一对一样本和任务映射的受约束多任务设置，忽视了多个任务可能在同一样本上操作的范式，例如场景理解。在本文中，我们关注单输入多输出（SIMO）的多任务设置，并展示了它在文献中研究的单输入单输出模型融合设置中存在质的差异，这是由于任务特定解码器和多样性的损失目标的存在。我们发现现有的模型融合方法导致显著的性能下降，主要原因是合并后的编码器和任务特定解码器之间的表示不一致。我们为SIMO设置提出了两种简单的高效解决方案，以合并后重新对齐特征表示。与联合微调相比，我们的方法在计算效率和灵活性方面更具优势，并有助于离线识别任务关系。实验在NYUv2、Cityscapes以及Taskonomy数据集的部分子集上证明：（1）任务算术足以使模型具备多任务能力；然而，合并后的编码器生成的表示需要重新与任务特定头对齐；（2）所提出的架构在性能上与传统多任务学习相当，但由于利用了特定任务模型的存在，所需的样本数量和训练步骤更少。 

---
# DeepSelective: Feature Gating and Representation Matching for Interpretable Clinical Prediction 

**Title (ZH)**: DeepSelective: 特征门控与表示匹配的可解释临床预测 

**Authors**: Ruochi Zhang, Qian Yang, Xiaoyang Wang, Haoran Wu, Qiong Zhou, Yu Wang, Kewei Li, Yueying Wang, Yusi Fan, Jiale Zhang, Lan Huang, Chang Liu, Fengfeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.11264)  

**Abstract**: The rapid accumulation of Electronic Health Records (EHRs) has transformed healthcare by providing valuable data that enhance clinical predictions and diagnoses. While conventional machine learning models have proven effective, they often lack robust representation learning and depend heavily on expert-crafted features. Although deep learning offers powerful solutions, it is often criticized for its lack of interpretability. To address these challenges, we propose DeepSelective, a novel end to end deep learning framework for predicting patient prognosis using EHR data, with a strong emphasis on enhancing model interpretability. DeepSelective combines data compression techniques with an innovative feature selection approach, integrating custom-designed modules that work together to improve both accuracy and interpretability. Our experiments demonstrate that DeepSelective not only enhances predictive accuracy but also significantly improves interpretability, making it a valuable tool for clinical decision-making. The source code is freely available at this http URL . 

**Abstract (ZH)**: 电子健康记录（EHR）的快速积累已通过提供增强临床预测和诊断的宝贵数据，彻底改变了医疗保健。尽管传统的机器学习模型已 proven 有效，但它们往往缺乏稳固的表示学习，并严重依赖专家设计的特征。虽然深度学习提供了强大的解决方案，但它常因缺乏可解释性而受到批评。为解决这些问题，我们提出了 DeepSelective——一种新颖的端到端深度学习框架，用于基于 EHR 数据预测患者预后，强调增强模型的可解释性。DeepSelective 结合了数据压缩技术与创新的功能选择方法，集成自定义设计的模块，共同提高准确性和可解释性。我们的实验表明，DeepSelective 不仅提高了预测准确性，还显著提升了可解释性，使其成为临床决策的重要工具。相关源代码已免费发布在该网址。 

---
# A Rollout-Based Algorithm and Reward Function for Efficient Resource Allocation in Business Processes 

**Title (ZH)**: 基于回放的算法及奖励函数在业务流程中高效资源分配 

**Authors**: Jeroen Middelhuis, Zaharah Bukhsh, Ivo Adan, Remco Dijkman  

**Link**: [PDF](https://arxiv.org/pdf/2504.11250)  

**Abstract**: Resource allocation plays a critical role in minimizing cycle time and improving the efficiency of business processes. Recently, Deep Reinforcement Learning (DRL) has emerged as a powerful tool to optimize resource allocation policies in business processes. In the DRL framework, an agent learns a policy through interaction with the environment, guided solely by reward signals that indicate the quality of its decisions. However, existing algorithms are not suitable for dynamic environments such as business processes. Furthermore, existing DRL-based methods rely on engineered reward functions that approximate the desired objective, but a misalignment between reward and objective can lead to undesired decisions or suboptimal policies. To address these issues, we propose a rollout-based DRL algorithm and a reward function to optimize the objective directly. Our algorithm iteratively improves the policy by evaluating execution trajectories following different actions. Our reward function directly decomposes the objective function of minimizing the mean cycle time. Maximizing our reward function guarantees that the objective function is minimized without requiring extensive reward engineering. The results show that our method consistently learns the optimal policy in all six evaluated business processes, outperforming the state-of-the-art algorithm that can only learn the optimal policy in two of the evaluated processes. 

**Abstract (ZH)**: 资源分配在最小化周期时间并提高业务流程效率中发挥着关键作用。近年来，深度强化学习（DRL）成为优化业务流程中资源分配策略的强大工具。在DRL框架中，智能体通过与环境的交互学习策略，仅受奖励信号的指导，这些信号表明其决策的质量。然而，现有算法不适用于如业务流程等动态环境。此外，现有的基于DRL的方法依赖于近似目标的工程化奖励函数，但奖励与目标之间的不匹配可能导致不良决策或次优策略。为解决这些问题，我们提出了一种基于展开的DRL算法和直接优化目标的奖励函数。该算法通过评估不同动作执行轨迹的策略逐步改进。我们的奖励函数直接将最小化平均周期时间的目标函数分解。最大化我们的奖励函数保证了目标函数的最小化，无需大量奖励工程。实验结果表明，我们的方法在六个评估的业务流程中始终学习到了最优策略，优于只能在两种评估流程中学习到最优策略的最先进算法。 

---
# Respiratory Inhaler Sound Event Classification Using Self-Supervised Learning 

**Title (ZH)**: 基于自监督学习的呼吸吸入器声音事件分类 

**Authors**: Davoud Shariat Panah, Alessandro N Franciosi, Cormac McCarthy, Andrew Hines  

**Link**: [PDF](https://arxiv.org/pdf/2504.11246)  

**Abstract**: Asthma is a chronic respiratory condition that affects millions of people worldwide. While this condition can be managed by administering controller medications through handheld inhalers, clinical studies have shown low adherence to the correct inhaler usage technique. Consequently, many patients may not receive the full benefit of their medication. Automated classification of inhaler sounds has recently been studied to assess medication adherence. However, the existing classification models were typically trained using data from specific inhaler types, and their ability to generalize to sounds from different inhalers remains unexplored. In this study, we adapted the wav2vec 2.0 self-supervised learning model for inhaler sound classification by pre-training and fine-tuning this model on inhaler sounds. The proposed model shows a balanced accuracy of 98% on a dataset collected using a dry powder inhaler and smartwatch device. The results also demonstrate that re-finetuning this model on minimal data from a target inhaler is a promising approach to adapting a generic inhaler sound classification model to a different inhaler device and audio capture hardware. This is the first study in the field to demonstrate the potential of smartwatches as assistive technologies for the personalized monitoring of inhaler adherence using machine learning models. 

**Abstract (ZH)**: 哮喘是一种影响全世界数百万人的慢性呼吸道疾病。尽管可以通过手持吸入器给药控制该疾病，但临床研究显示患者正确使用吸入器的技术 adherence 较低，这可能导致患者未能充分发挥药物疗效。最近，吸入器声音的自动化分类已被研究以评估药物依从性。然而，现有的分类模型通常仅在特定吸入器类型的数据上进行训练，其在不同吸入器类型上通用性的能力尚未得到探索。在本研究中，我们通过在吸入器声音上进行预训练和微调来适应 wav2vec 2.0 自监督学习模型进行吸入器声音分类。提出的新模型在使用干粉吸入器和智能手表设备收集的数据集上达到了 98% 的平衡准确率。研究结果还表明，通过少量目标吸入器数据对模型进行再微调是将通用的吸入器声音分类模型适应到不同吸入器设备和音频采集硬件的一种有前景的方法。这是该领域首项展示了智能手表作为机器学习模型辅助技术，用于个性化监测吸入器依从性的潜力的研究。 

---
# Influence Maximization in Temporal Social Networks with a Cold-Start Problem: A Supervised Approach 

**Title (ZH)**: 冷启动问题下Temporal社交网络中的影响力最大化：一种监督方法 

**Authors**: Laixin Xie, Ying Zhang, Xiyuan Wang, Shiyi Liu, Shenghan Gao, Xingxing Xing, Wei Wan, Haipeng Zhang, Quan Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11245)  

**Abstract**: Influence Maximization (IM) in temporal graphs focuses on identifying influential "seeds" that are pivotal for maximizing network expansion. We advocate defining these seeds through Influence Propagation Paths (IPPs), which is essential for scaling up the network. Our focus lies in efficiently labeling IPPs and accurately predicting these seeds, while addressing the often-overlooked cold-start issue prevalent in temporal networks. Our strategy introduces a motif-based labeling method and a tensorized Temporal Graph Network (TGN) tailored for multi-relational temporal graphs, bolstering prediction accuracy and computational efficiency. Moreover, we augment cold-start nodes with new neighbors from historical data sharing similar IPPs. The recommendation system within an online team-based gaming environment presents subtle impact on the social network, forming multi-relational (i.e., weak and strong) temporal graphs for our empirical IM study. We conduct offline experiments to assess prediction accuracy and model training efficiency, complemented by online A/B testing to validate practical network growth and the effectiveness in addressing the cold-start issue. 

**Abstract (ZH)**: 时效网络中的影响力最大化（IM）旨在识别对网络扩展至关重要的“种子”节点。我们提倡通过影响传播路径（IPPs）来定义这些种子节点，这对于扩展网络至关重要。我们的重点在于高效地标记IPPs并准确预测这些种子节点，同时应对时效网络中普遍存在的冷启动问题。我们提出的策略包括基于模式的标记方法和适用于多关系时效图的张量化时效图网络（TGN），这提高了预测准确性和计算效率。此外，我们通过从历史数据中添加具有相似IPP的新邻居来增强冷启动节点。在线团队游戏环境中推荐系统的影响力在社会网络中表现微妙，形成了多关系（即弱关系和强关系）时效图，用于我们的实证IM研究。我们进行离线实验评估预测准确性和模型训练效率，并结合在线A/B测试验证实际网络扩展的有效性和解决冷启动问题的实用性。 

---
# Divergence of Empirical Neural Tangent Kernel in Classification Problems 

**Title (ZH)**: 分类问题中经验神经 tangent 核的发散性 

**Authors**: Zixiong Yu, Songtao Tian, Guhan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11130)  

**Abstract**: This paper demonstrates that in classification problems, fully connected neural networks (FCNs) and residual neural networks (ResNets) cannot be approximated by kernel logistic regression based on the Neural Tangent Kernel (NTK) under overtraining (i.e., when training time approaches infinity). Specifically, when using the cross-entropy loss, regardless of how large the network width is (as long as it is finite), the empirical NTK diverges from the NTK on the training samples as training time increases. To establish this result, we first demonstrate the strictly positive definiteness of the NTKs for multi-layer FCNs and ResNets. Then, we prove that during training, % with the cross-entropy loss, the neural network parameters diverge if the smallest eigenvalue of the empirical NTK matrix (Gram matrix) with respect to training samples is bounded below by a positive constant. This behavior contrasts sharply with the lazy training regime commonly observed in regression problems. Consequently, using a proof by contradiction, we show that the empirical NTK does not uniformly converge to the NTK across all times on the training samples as the network width increases. We validate our theoretical results through experiments on both synthetic data and the MNIST classification task. This finding implies that NTK theory is not applicable in this context, with significant theoretical implications for understanding neural networks in classification problems. 

**Abstract (ZH)**: 这篇论文展示了在分类问题中，完全连接神经网络（FCNs）和残差神经网络（ResNets）在过拟合情况下（即训练时间趋近于无穷大）不能被神经领域核（NTK）基于核逻辑回归逼近。具体而言，使用交叉熵损失函数时，无论网络宽度多大（只要不是无限大），训练时间增加时经验NTK将远离训练样本上的NTK。为了得出这一结果，我们首先证明了多层FCNs和ResNets的NTK严格正定性。然后我们证明，在使用交叉熵损失函数训练过程中，如果训练样本的经验NTK矩阵（Gram矩阵）的最小特征值被正常数下界限制，则神经网络参数会发散。这种行为与回归问题中常见的惰性训练阶段形成了鲜明对比。通过反证法，我们证明随着网络宽度的增加，经验NTK在训练样本上不一致地收敛到NTK。通过在合成数据和MNIST分类任务上的实验验证了我们的理论结果。这一发现表明NTK理论在此上下文中不适用，对理解分类问题中的神经网络具有重要的理论意义。 

---
# AI-guided Antibiotic Discovery Pipeline from Target Selection to Compound Identification 

**Title (ZH)**: AI引导的从靶标选择到化合物鉴定的抗生素发现流程 

**Authors**: Maximilian G. Schuh, Joshua Hesse, Stephan A. Sieber  

**Link**: [PDF](https://arxiv.org/pdf/2504.11091)  

**Abstract**: Antibiotic resistance presents a growing global health crisis, demanding new therapeutic strategies that target novel bacterial mechanisms. Recent advances in protein structure prediction and machine learning-driven molecule generation offer a promising opportunity to accelerate drug discovery. However, practical guidance on selecting and integrating these models into real-world pipelines remains limited. In this study, we develop an end-to-end, artificial intelligence-guided antibiotic discovery pipeline that spans target identification to compound realization. We leverage structure-based clustering across predicted proteomes of multiple pathogens to identify conserved, essential, and non-human-homologous targets. We then systematically evaluate six leading 3D-structure-aware generative models$\unicode{x2014}$spanning diffusion, autoregressive, graph neural network, and language model architectures$\unicode{x2014}$on their usability, chemical validity, and biological relevance. Rigorous post-processing filters and commercial analogue searches reduce over 100 000 generated compounds to a focused, synthesizable set. Our results highlight DeepBlock and TamGen as top performers across diverse criteria, while also revealing critical trade-offs between model complexity, usability, and output quality. This work provides a comparative benchmark and blueprint for deploying artificial intelligence in early-stage antibiotic development. 

**Abstract (ZH)**: 抗生素耐药性构成了日益严峻的全球健康危机，需要针对新型细菌机制的新治疗策略。近年来，基于蛋白质结构预测和机器学习驱动的分子生成技术为加速药物发现提供了新的机会。然而，关于如何选择和将这些模型整合到实际工作流程中的实用指导仍显不足。在本研究中，我们开发了一个从靶标识别到化合物实现的端到端、人工智能引导的抗生素发现管道。我们利用针对多种病原体预测蛋白质组的结构基团聚类来识别保守、必不可少且非人类同源的靶标。然后，我们系统地评估了六种领先的空间结构感知生成模型——包括扩散模型、自回归模型、图神经网络和语言模型架构——的可使用性、化学合理性和生物学相关性。严格的后处理过滤和商业同系物搜索将超过100,000个生成化合物缩减为一个集中且可合成的集合。我们的结果强调了DeepBlock和TamGen在多种标准下的表现最佳，同时也揭示了模型复杂性、可使用性和输出质量之间的关键权衡。本研究提供了人工智能在早期抗生素开发中应用的比较基准和蓝图。 

---
# Dynamical errors in machine learning forecasts 

**Title (ZH)**: 机器学习预测中的动态误差 

**Authors**: Zhou Fang, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2504.11074)  

**Abstract**: In machine learning forecasting, standard error metrics such as mean absolute error (MAE) and mean squared error (MSE) quantify discrepancies between predictions and target values. However, these metrics do not directly evaluate the physical and/or dynamical consistency of forecasts, an increasingly critical concern in scientific and engineering applications.
Indeed, a fundamental yet often overlooked question is whether machine learning forecasts preserve the dynamical behavior of the underlying system. Addressing this issue is essential for assessing the fidelity of machine learning models and identifying potential failure modes, particularly in applications where maintaining correct dynamical behavior is crucial.
In this work, we investigate the relationship between standard forecasting error metrics, such as MAE and MSE, and the dynamical properties of the underlying system. To achieve this goal, we use two recently developed dynamical indices: the instantaneous dimension ($d$), and the inverse persistence ($\theta$). Our results indicate that larger forecast errors -- e.g., higher MSE -- tend to occur in states with higher $d$ (higher complexity) and higher $\theta$ (lower persistence). To further assess dynamical consistency, we propose error metrics based on the dynamical indices that measure the discrepancy of the forecasted $d$ and $\theta$ versus their correct values. Leveraging these dynamical indices-based metrics, we analyze direct and recursive forecasting strategies for three canonical datasets -- Lorenz, Kuramoto-Sivashinsky equation, and Kolmogorov flow -- as well as a real-world weather forecasting task. Our findings reveal substantial distortions in dynamical properties in ML forecasts, especially for long forecast lead times or long recursive simulations, providing complementary information on ML forecast fidelity that can be used to improve ML models. 

**Abstract (ZH)**: 在机器学习预测中，标准误差指标如均方误差（MSE）和均绝对误差（MAE）量度了预测值与目标值之间的差异，但这些指标没有直接评估 forecasts 的物理和/或动力学一致性，这对于科学和工程应用来说越来越成为一个关键问题。实际上，一个基础但常被忽视的问题是机器学习预测是否保留了底层系统的动力学行为。解决这一问题对于评估机器学习模型的保真度和识别潜在故障模式至关重要，特别是在必须保持正确动力学行为的应用中。在本文中，我们研究了标准预测误差指标（如 MSE 和 MAE）与底层系统动力学属性之间的关系。为此，我们使用了两个 recently 开发的动力学指标：瞬时维度 ($d$) 和逆持续性 ($\theta$)。结果显示，较大的预测误差（例如，较高的 MSE）倾向于出现在更高 $d$（更高复杂度）和更高 $\theta$（更低持续性）的状态中。为了进一步评估动力学一致性，我们提出了基于动力学指标的误差指标，这些指标测量了 forecasted $d$ 和 $\theta$ 与其正确值之间的差异。利用这些基于动力学指标的指标，我们分析了三维标准数据集（洛伦兹系统、库拉莫托-西凡夏系统方程和柯尔莫哥洛夫流动）以及实际天气预报任务的直接和递归预报策略。我们的发现揭示了在机器学习预测中动力学属性的显著失真，尤其是在较长预测时效或较长递归模拟中，提供了关于机器学习预测保真度的补充信息，可用于改进机器学习模型。 

---
# QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models 

**Title (ZH)**: QAVA：面向大型视觉-语言模型的查询无感知视觉攻击 

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11038)  

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at this https URL. 

**Abstract (ZH)**: 在典型的多模态任务中，如视觉问答（VQA），针对特定图像和问题的对抗攻击可以使大型视听模型（LVLMs）提供错误的答案。然而，一张图片通常与多个问题相关联，即使图片被针对特定问题进行了攻击，LVLMs也可能仍能正确回答其他问题。为了应对这一挑战，我们引入了一种查询无感知的视觉攻击（QAVA），其目标是生成对未指定和未知问题产生错误响应的健壯对抗样本。与传统针对特定图像和问题的对抗攻击相比，QAVA在未知问题的背景下增强了针对图片的攻击效果，其性能与针对已知目标问题的攻击相当。我们的研究拓宽了在实际应用场景中LVLMs的视觉对抗攻击范围，揭示了之前忽视的脆弱性，特别是在视觉对抗威胁的背景下。代码可在此处获得：this https URL。 

---
# "Even explanations will not help in trusting [this] fundamentally biased system": A Predictive Policing Case-Study 

**Title (ZH)**: “即使解释也无法让人信任这一基本有偏见的系统”：一项预测性警务案例研究 

**Authors**: Siddharth Mehrotra, Ujwal Gadiraju, Eva Bittner, Folkert van Delden, Catholijn M. Jonker, Myrthe L. Tielman  

**Link**: [PDF](https://arxiv.org/pdf/2504.11020)  

**Abstract**: In today's society, where Artificial Intelligence (AI) has gained a vital role, concerns regarding user's trust have garnered significant attention. The use of AI systems in high-risk domains have often led users to either under-trust it, potentially causing inadequate reliance or over-trust it, resulting in over-compliance. Therefore, users must maintain an appropriate level of trust. Past research has indicated that explanations provided by AI systems can enhance user understanding of when to trust or not trust the system. However, the utility of presentation of different explanations forms still remains to be explored especially in high-risk domains. Therefore, this study explores the impact of different explanation types (text, visual, and hybrid) and user expertise (retired police officers and lay users) on establishing appropriate trust in AI-based predictive policing. While we observed that the hybrid form of explanations increased the subjective trust in AI for expert users, it did not led to better decision-making. Furthermore, no form of explanations helped build appropriate trust. The findings of our study emphasize the importance of re-evaluating the use of explanations to build [appropriate] trust in AI based systems especially when the system's use is questionable. Finally, we synthesize potential challenges and policy recommendations based on our results to design for appropriate trust in high-risk based AI-based systems. 

**Abstract (ZH)**: 人工智能基于预测警务中不同解释类型和用户专业背景对建立适当信任的影响研究 

---
# Document Quality Scoring for Web Crawling 

**Title (ZH)**: 网页抓取中的文档质量评分 

**Authors**: Francesca Pezzuti, Ariane Mueller, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11011)  

**Abstract**: The internet contains large amounts of low-quality content, yet users expect web search engines to deliver high-quality, relevant results. The abundant presence of low-quality pages can negatively impact retrieval and crawling processes by wasting resources on these documents. Therefore, search engines can greatly benefit from techniques that leverage efficient quality estimation methods to mitigate these negative impacts. Quality scoring methods for web pages are useful for many processes typical for web search systems, including static index pruning, index tiering, and crawling. Building on work by Chang et al.~\cite{chang2024neural}, who proposed using neural estimators of semantic quality for static index pruning, we extend their approach and apply their neural quality scorers to assess the semantic quality of web pages in crawling prioritisation tasks. In our experimental analysis, we found that prioritising semantically high-quality pages over low-quality ones can improve downstream search effectiveness. Our software contribution consists of a Docker container that computes an effective quality score for a given web page, allowing the quality scorer to be easily included and used in other components of web search systems. 

**Abstract (ZH)**: 互联网包含大量低质量内容，但用户期望网络搜索引擎能够提供高质量的相关结果。大量低质量页面的存在会对检索和爬虫过程产生负面影响，浪费资源在这些文档上。因此，搜索引擎可以从利用高效的质量估计方法中获益，以减轻这些负面影响。用于网页的质量评分方法对网页搜索系统中的许多典型过程（包括静态索引修剪、索引分层和爬虫）很有用。在此基础上，我们借鉴Chang等人的工作（Chang et al.~\cite{chang2024neural}），提出使用神经估计器对静态索引进行修剪，并进一步将他们的神经质量评分器应用于爬虫优先级任务中以评估网页的语义质量。实验分析表明，优先处理语义质量高的网页可以提高后续搜索效果。我们的软件贡献在于提供一个Docker容器，用于计算给定网页的有效质量得分，使得质量评分器可以轻松地被纳入和使用于网页搜索系统的其他组件中。 

---
# ProtFlow: Fast Protein Sequence Design via Flow Matching on Compressed Protein Language Model Embeddings 

**Title (ZH)**: ProtFlow: 快速蛋白质序列设计通过压缩蛋白质语言模型嵌入的流匹配 

**Authors**: Zitai Kong, Yiheng Zhu, Yinlong Xu, Hanjing Zhou, Mingzhe Yin, Jialu Wu, Hongxia Xu, Chang-Yu Hsieh, Tingjun Hou, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10983)  

**Abstract**: The design of protein sequences with desired functionalities is a fundamental task in protein engineering. Deep generative methods, such as autoregressive models and diffusion models, have greatly accelerated the discovery of novel protein sequences. However, these methods mainly focus on local or shallow residual semantics and suffer from low inference efficiency, large modeling space and high training cost. To address these challenges, we introduce ProtFlow, a fast flow matching-based protein sequence design framework that operates on embeddings derived from semantically meaningful latent space of protein language models. By compressing and smoothing the latent space, ProtFlow enhances performance while training on limited computational resources. Leveraging reflow techniques, ProtFlow enables high-quality single-step sequence generation. Additionally, we develop a joint design pipeline for the design scene of multichain proteins. We evaluate ProtFlow across diverse protein design tasks, including general peptides and long-chain proteins, antimicrobial peptides, and antibodies. Experimental results demonstrate that ProtFlow outperforms task-specific methods in these applications, underscoring its potential and broad applicability in computational protein sequence design and analysis. 

**Abstract (ZH)**: 基于语义有意义潜在空间的流匹配蛋白序列设计框架 

---
# Exploring the Role of KG-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs 

**Title (ZH)**: 基于知识图谱的RAG在小型LLM驱动的日语医疗问答中的作用探索 

**Authors**: Yingjian Chen, Feiyang Li, Xingyu Song, Tianxiao Li, Issey Sudeka, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10982)  

**Abstract**: Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages. 

**Abstract (ZH)**: 基于知识图谱的检索增强生成框架(KG-RAG)在小型开源日医QA中的探索 

---
# Evaluating Trust in AI, Human, and Co-produced Feedback Among Undergraduate Students 

**Title (ZH)**: 评估本科生在AI、人类和联合生产反馈中的信任度 

**Authors**: Audrey Zhang, Yifei Gao, Wannapon Suraworachet, Tanya Nazaretsky, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2504.10961)  

**Abstract**: As generative AI transforms educational feedback practices, understanding students' perceptions of different feedback providers becomes crucial for effective implementation. This study addresses a critical gap by comparing undergraduate students' trust in AI-generated, human-created, and human-AI co-produced feedback, informing how institutions can adapt feedback practices in this new era. Through a within-subject experiment with 91 participants, we investigated factors predicting students' ability to distinguish between feedback types, perception of feedback quality, and potential biases to AI involvement. Findings revealed that students generally preferred AI and co-produced feedback over human feedback in terms of perceived usefulness and objectivity. Only AI feedback suffered a decline in perceived genuineness when feedback sources were revealed, while co-produced feedback maintained its positive perception. Educational AI experience improved students' ability to identify AI feedback and increased their trust in all feedback types, while general AI experience decreased perceived usefulness and credibility. Male students consistently rated all feedback types as less valuable than their female and non-binary counterparts. These insights inform evidence-based guidelines for integrating AI into higher education feedback systems while addressing trust concerns and fostering AI literacy among students. 

**Abstract (ZH)**: 随着生成式AI改变教育反馈实践，理解不同反馈提供者的学生感知成为有效实施的关键。本研究通过对比大学生对AI生成、人类创造和人类与AI合作生成反馈的信任度，填补了重要空白，并为机构如何适应这一新环境下的反馈实践提供信息。通过一项涉及91名参与者的重复被试实验，我们探讨了影响学生区分不同反馈类型能力、反馈质量感知以及对AI参与可能存在的偏见的因素。研究发现，学生普遍更偏好AI和合作生成的反馈，认为其更具实用性和客观性。只有当反馈来源被揭示时，AI反馈的可信度有所下降，而合作生成的反馈维持了积极的感知。教育AI经验提高了学生识别AI反馈的能力，并增加了他们对所有反馈类型的信任，而一般AI经验则降低了反馈的实用性和可信度。男性学生普遍认为所有反馈类型的价值低于女性和非二元性别学生。这些见解为基于证据的指南提供了依据，指导AI整合入高等教育反馈系统，并解决信任关切，培养学生的AI素养。 

---
# BEACON: A Benchmark for Efficient and Accurate Counting of Subgraphs 

**Title (ZH)**: BEACON: 一个用于子图高效准确计数的基准测试 

**Authors**: Mohammad Matin Najafi, Xianju Zhu, Chrysanthi Kosyfaki, Laks V.S. Lakshmanan, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10948)  

**Abstract**: Subgraph counting the task of determining the number of instances of a query pattern within a large graph lies at the heart of many critical applications, from analyzing financial networks and transportation systems to understanding biological interactions. Despite decades of work yielding efficient algorithmic (AL) solutions and, more recently, machine learning (ML) approaches, a clear comparative understanding is elusive. This gap stems from the absence of a unified evaluation framework, standardized datasets, and accessible ground truths, all of which hinder systematic analysis and fair benchmarking. To overcome these barriers, we introduce BEACON: a comprehensive benchmark designed to rigorously evaluate both AL and ML-based subgraph counting methods. BEACON provides a standardized dataset with verified ground truths, an integrated evaluation environment, and a public leaderboard, enabling reproducible and transparent comparisons across diverse approaches. Our extensive experiments reveal that while AL methods excel in efficiently counting subgraphs on very large graphs, they struggle with complex patterns (e.g., those exceeding six nodes). In contrast, ML methods are capable of handling larger patterns but demand massive graph data inputs and often yield suboptimal accuracy on small, dense graphs. These insights not only highlight the unique strengths and limitations of each approach but also pave the way for future advancements in subgraph counting techniques. Overall, BEACON represents a significant step towards unifying and accelerating research in subgraph counting, encouraging innovative solutions and fostering a deeper understanding of the trade-offs between algorithmic and machine learning paradigms. 

**Abstract (ZH)**: BEACON：一种用于子图计数方法综合评估的基准框架 

---
# Transfer Learning for Temporal Link Prediction 

**Title (ZH)**: 时间链接预测中的迁移学习 

**Authors**: Ayan Chatterjee, Barbara Ikica, Babak Ravandi, John Palowitch  

**Link**: [PDF](https://arxiv.org/pdf/2504.10925)  

**Abstract**: Link prediction on graphs has applications spanning from recommender systems to drug discovery. Temporal link prediction (TLP) refers to predicting future links in a temporally evolving graph and adds additional complexity related to the dynamic nature of graphs. State-of-the-art TLP models incorporate memory modules alongside graph neural networks to learn both the temporal mechanisms of incoming nodes and the evolving graph topology. However, memory modules only store information about nodes seen at train time, and hence such models cannot be directly transferred to entirely new graphs at test time and deployment. In this work, we study a new transfer learning task for temporal link prediction, and develop transfer-effective methods for memory-laden models. Specifically, motivated by work showing the informativeness of structural signals for the TLP task, we augment a structural mapping module to the existing TLP model architectures, which learns a mapping from graph structural (topological) features to memory embeddings. Our work paves the way for a memory-free foundation model for TLP. 

**Abstract (ZH)**: 图上的链接预测在从推荐系统到药物发现等多个领域都有应用。时间链接预测（TLP）是指预测时间演变图中的未来链接，并且增加了与图的动态性质相关的额外复杂性。最先进的TLP模型结合了记忆模块和图神经网络，以学习入边节点的时间机制以及图拓扑的演变。然而，记忆模块只能存储训练期间看到的节点信息，因此此类模型无法直接迁移应用于测试时间和部署中的全新图。在这项工作中，我们研究了时间链接预测的新型迁移学习任务，并开发了适用于记忆负载模型的有效迁移方法。具体来说，受结构信号对于TLP任务有用性的研究启发，我们在现有TLP模型架构中增加了一个结构映射模块，该模块学习从图结构（拓扑）特征到记忆嵌入的映射。我们的工作为TLP奠定了无记忆基础模型的道路。 

---
# Towards A Universal Graph Structural Encoder 

**Title (ZH)**: 面向通用图结构编码器 

**Authors**: Jialin Chen, Haolan Zuo, Haoyu Peter Wang, Siqi Miao, Pan Li, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2504.10917)  

**Abstract**: Recent advancements in large-scale pre-training have shown the potential to learn generalizable representations for downstream tasks. In the graph domain, however, capturing and transferring structural information across different graph domains remains challenging, primarily due to the inherent differences in topological patterns across various contexts. Additionally, most existing models struggle to capture the complexity of rich graph structures, leading to inadequate exploration of the embedding space. To address these challenges, we propose GFSE, a universal graph structural encoder designed to capture transferable structural patterns across diverse domains such as molecular graphs, social networks, and citation networks. GFSE is the first cross-domain graph structural encoder pre-trained with multiple self-supervised learning objectives. Built on a Graph Transformer, GFSE incorporates attention mechanisms informed by graph inductive bias, enabling it to encode intricate multi-level and fine-grained topological features. The pre-trained GFSE produces generic and theoretically expressive positional and structural encoding for graphs, which can be seamlessly integrated with various downstream graph feature encoders, including graph neural networks for vectorized features and Large Language Models for text-attributed graphs. Comprehensive experiments on synthetic and real-world datasets demonstrate GFSE's capability to significantly enhance the model's performance while requiring substantially less task-specific fine-tuning. Notably, GFSE achieves state-of-the-art performance in 81.6% evaluated cases, spanning diverse graph models and datasets, highlighting its potential as a powerful and versatile encoder for graph-structured data. 

**Abstract (ZH)**: 近年来，大规模预训练的最新进展展示了学习下游任务泛化表示的潜力。然而，在图领域中，跨不同图域捕捉和传递结构信息依然充满挑战，主要原因是各种上下文之间的拓扑模式存在固有的差异。此外，大多数现有模型难以捕捉复杂图结构的细节，导致嵌入空间的探索不够充分。为了解决这些问题，我们提出了一种名为GFSE的通用图结构编码器，旨在跨分子图、社交网络和引用网络等不同领域捕捉可转移的结构模式。GFSE是首个基于多种自监督学习目标预训练的跨域图结构编码器。基于图变压器，GFSE融合了由图归纳偏置驱动的注意力机制，使其能够编码复杂的多层次和精细的拓扑特征。预训练的GFSE生成适用于各类下游图特征编码器的通用且理论表达性强的位置编码和结构编码，包括用于向量特征的图神经网络和用于文本图的大型语言模型。综合实验表明，GFSE能够在合成和真实世界数据集上显著提升模型性能，同时需要较少的任务特定微调。值得注意的是，GFSE在81.6%的评估案例中达到了最先进的性能，涵盖了多种图模型和数据集，突显了其作为图结构数据强大而通用编码器的潜力。 

---
# LOKA Protocol: A Decentralized Framework for Trustworthy and Ethical AI Agent Ecosystems 

**Title (ZH)**: LOKA协议：一个可信赖和伦理的AI代理生态系统去中心化框架 

**Authors**: Rajesh Ranjan, Shailja Gupta, Surya Narayan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2504.10915)  

**Abstract**: The rise of autonomous AI agents, capable of perceiving, reasoning, and acting independently, signals a profound shift in how digital ecosystems operate, govern, and evolve. As these agents proliferate beyond centralized infrastructures, they expose foundational gaps in identity, accountability, and ethical alignment. Three critical questions emerge: Identity: Who or what is the agent? Accountability: Can its actions be verified, audited, and trusted? Ethical Consensus: Can autonomous systems reliably align with human values and prevent harmful emergent behaviors? We present the novel LOKA Protocol (Layered Orchestration for Knowledgeful Agents), a unified, systems-level architecture for building ethically governed, interoperable AI agent ecosystems. LOKA introduces a proposed Universal Agent Identity Layer (UAIL) for decentralized, verifiable identity; intent-centric communication protocols for semantic coordination across diverse agents; and a Decentralized Ethical Consensus Protocol (DECP) that enables agents to make context-aware decisions grounded in shared ethical baselines. Anchored in emerging standards such as Decentralized Identifiers (DIDs), Verifiable Credentials (VCs), and post-quantum cryptography, LOKA offers a scalable, future-resilient blueprint for multi-agent AI governance. By embedding identity, trust, and ethics into the protocol layer itself, LOKA establishes the foundation for a new era of responsible, transparent, and autonomous AI ecosystems operating across digital and physical domains. 

**Abstract (ZH)**: 自主AI代理的兴起标志着数字生态系统运作、治理和演化的深刻转变。随着这些代理超越集中式基础设施的普及，它们暴露了身份、问责制和伦理对齐的基本缺口。三个关键问题随之浮现：身份：代理是什么？问责制：其行动能否被验证、审计并信任？伦理一致性：自主系统能否可靠地与人类价值观对齐并防止有害的 emergent 行为？我们提出了新型LOKA协议（知识型代理的分层编排），这是一种统一的系统级架构，用于构建受伦理治理和互联互通的AI代理生态系统。LOKA引入了提议的去中心化可验证身份层（UAIL）、以意图为中心的通信协议以实现跨不同代理的语义协调，以及去中心化的伦理一致性协议（DECP），使代理能够基于共享的伦理基准做出情境感知的决策。基于如去中心化标识符（DIDs）、可验证凭据（VCs）和后量子加密等新兴标准，LOKA为多代理AI治理提供了可扩展且面向未来的蓝图。通过将身份、信任和伦理嵌入协议层本身，LOKA为横跨数字和物理域的责任、透明和自主AI生态系统的时代奠定了基础。 

---
# Efficient Reasoning Models: A Survey 

**Title (ZH)**: 高效推理模型：一个综述 

**Authors**: Sicheng Feng, Gongfan Fang, Xinyin Ma, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10903)  

**Abstract**: Reasoning models have demonstrated remarkable progress in solving complex and logic-intensive tasks by generating extended Chain-of-Thoughts (CoTs) prior to arriving at a final answer. Yet, the emergence of this "slow-thinking" paradigm, with numerous tokens generated in sequence, inevitably introduces substantial computational overhead. To this end, it highlights an urgent need for effective acceleration. This survey aims to provide a comprehensive overview of recent advances in efficient reasoning. It categorizes existing works into three key directions: (1) shorter - compressing lengthy CoTs into concise yet effective reasoning chains; (2) smaller - developing compact language models with strong reasoning capabilities through techniques such as knowledge distillation, other model compression techniques, and reinforcement learning; and (3) faster - designing efficient decoding strategies to accelerate inference. A curated collection of papers discussed in this survey is available in our GitHub repository. 

**Abstract (ZH)**: 推理模型已经在通过生成扩展的思维链(CoTs)来解决复杂和逻辑密集型任务方面取得了显著进展，但在到达最终答案之前生成这些思维链不可避免地带来了大量的计算开销。为此，有效加速显得尤为迫切。本文综述旨在提供对近期高效推理进展的全面概述。它将现有工作划分为三个主要方向：(1) 更短——将长思维链压缩为简洁有效的推理链；(2) 更小——通过知识蒸馏、其他模型压缩技术及强化学习等手段开发具有强大推理能力的紧凑型语言模型；(3) 更快——设计高效的解码策略以加速推理。本文综述中讨论的精选论文集合可在我们的GitHub仓库中获取。 

---
# Bridging Distribution Gaps in Time Series Foundation Model Pretraining with Prototype-Guided Normalization 

**Title (ZH)**: 基于原型引导归一化的时序基础模型预训练中分布差距弥合 

**Authors**: Peiliang Gong, Emadeldeen Eldele, Min Wu, Zhenghua Chen, Xiaoli Li, Daoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10900)  

**Abstract**: Foundation models have achieved remarkable success across diverse machine-learning domains through large-scale pretraining on large, diverse datasets. However, pretraining on such datasets introduces significant challenges due to substantial mismatches in data distributions, a problem particularly pronounced with time series data. In this paper, we tackle this issue by proposing a domain-aware adaptive normalization strategy within the Transformer architecture. Specifically, we replace the traditional LayerNorm with a prototype-guided dynamic normalization mechanism (ProtoNorm), where learned prototypes encapsulate distinct data distributions, and sample-to-prototype affinity determines the appropriate normalization layer. This mechanism effectively captures the heterogeneity of time series characteristics, aligning pretrained representations with downstream tasks. Through comprehensive empirical evaluation, we demonstrate that our method significantly outperforms conventional pretraining techniques across both classification and forecasting tasks, while effectively mitigating the adverse effects of distribution shifts during pretraining. Incorporating ProtoNorm is as simple as replacing a single line of code. Extensive experiments on diverse real-world time series benchmarks validate the robustness and generalizability of our approach, advancing the development of more versatile time series foundation models. 

**Abstract (ZH)**: 基于Transformer架构的原型引导自适应归一化策略：提升时间序列数据预训练效果 

---
# Xpose: Bi-directional Engineering for Hidden Query Extraction 

**Title (ZH)**: Xpose: 双向工程提取隐藏查询 

**Authors**: Ahana Pradhan, Jayant Haritsa  

**Link**: [PDF](https://arxiv.org/pdf/2504.10898)  

**Abstract**: Query reverse engineering (QRE) aims to synthesize a SQL query to connect a given database and result instance. A recent variation of QRE is where an additional input, an opaque executable containing a ground-truth query, is provided, and the goal is to non-invasively extract this specific query through only input-output examples. This variant, called Hidden Query Extraction (HQE), has a spectrum of industrial use-cases including query recovery, database security, and vendor migration. The reverse engineering (RE) tools developed for HQE, which are based on database mutation and generation techniques, can only extract flat queries with key-based equi joins and conjunctive arithmetic filter predicates, making them limited wrt both query structure and query operators. In this paper, we present Xpose, a HQE solution that elevates the extraction scope to realistic complex queries, such as those found in the TPCH benchmark. A two-pronged approach is taken: (1) The existing RE scope is substantially extended to incorporate union connectors, algebraic filter predicates, and disjunctions for both values and predicates. (2) The predictive power of LLMs is leveraged to convert business descriptions of the opaque application into extraction guidance, representing ``forward engineering" (FE). The FE module recognizes common constructs, such as nesting of sub-queries, outer joins, and scalar functions. In essence, FE establishes the broad query contours, while RE fleshes out the fine-grained details. We have evaluated Xpose on (a) E-TPCH, a query suite comprising the complete TPCH benchmark extended with queries featuring unions, diverse join types, and sub-queries; and (b) the real-world STACK benchmark. The experimental results demonstrate that its bi-directional engineering approach accurately extracts these complex queries, representing a significant step forward with regard to HQE coverage. 

**Abstract (ZH)**: 隐查询提取 (HQE) 旨在合成一个 SQL 查询以连接给定的数据库和结果实例。HQE 的一种变体提供了额外的输入，即包含真实查询的不透明可执行文件，并通过输入输出示例无侵入地提取该特定查询。该变体称为隐查询提取 (HQE)，其在查询恢复、数据库安全和供应商迁移等领域具有广泛的应用场景。为 HQE 开发的基于数据库变异和生成技术的逆向工程 (RE) 工具仅能提取基于键的等值连接和平面查询，且含有连接和算术过滤谓词，这使它们在查询结构和查询操作符方面都受到限制。在这篇论文中，我们提出了 Xpose，一个将提取范围扩展到现实中的复杂查询（如 TPCH 基准中的查询）的 HQE 解决方案。我们采用了双管齐下的方法：(1) 显著扩展现有 RE 范围，以结合并连接连接符、代数过滤谓词以及值和谓词的析取。(2) 利用大型语言模型 (LLM) 的预测能力，将不透明应用程序的业务描述转换为提取指导，代表了“正向工程”(FE)。FE 模块识别常见的构造，如子查询的嵌套、外连接和标量函数。本质上，FE 确定了广泛的查询轮廓，而 RE 填充了细粒度的细节。我们已在 (a) E-TPCH，一个包含完整 TPCH 基准并扩展了具有并集、多种连接类型和子查询的查询的查询集；和 (b) 实际的 STACK 基准上评估了 Xpose。实验结果表明，其双向工程方法准确提取了这些复杂查询，标志着 HQE 覆盖范围的一个重要进展。 

---
# Uplink Assisted Joint Channel Estimation and CSI Feedback: An Approach Based on Deep Joint Source-Channel Coding 

**Title (ZH)**: 上行协助联合信道估计和CSI反馈：基于深度联合源-信道编码的方法 

**Authors**: Yiran Guo, Wei Chen, Bo Ai  

**Link**: [PDF](https://arxiv.org/pdf/2504.10836)  

**Abstract**: In frequency division duplex (FDD) multiple-input multiple-output (MIMO) wireless communication systems, the acquisition of downlink channel state information (CSI) is essential for maximizing spatial resource utilization and improving system spectral efficiency. The separate design of modules in AI-based CSI feedback architectures under traditional modular communication frameworks, including channel estimation (CE), CSI compression and feedback, leads to sub-optimal performance. In this paper, we propose an uplink assisted joint CE and and CSI feedback approach via deep learning for downlink CSI acquisition, which mitigates performance degradation caused by distribution bias across separately trained modules in traditional modular communication frameworks. The proposed network adopts a deep joint source-channel coding (DJSCC) architecture to mitigate the cliff effect encountered in the conventional separate source-channel coding. Furthermore, we exploit the uplink CSI as auxiliary information to enhance CSI reconstruction accuracy by leveraging the partial reciprocity between the uplink and downlink channels in FDD systems, without introducing additional overhead. The effectiveness of uplink CSI as assisted information and the necessity of an end-toend multi-module joint training architecture is validated through comprehensive ablation and scalability experiments. 

**Abstract (ZH)**: 基于深度学习的上行辅助联合信道估计与CSI反馈方法在FDD MIMO无线通信系统中的下行CSI获取 

---
# Towards Spatially-Aware and Optimally Faithful Concept-Based Explanations 

**Title (ZH)**: 面向空间意识和最优忠实概念导向的解释 

**Authors**: Shubham Kumar, Dwip Dalal, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2504.10833)  

**Abstract**: Post-hoc, unsupervised concept-based explanation methods (U-CBEMs) are a promising tool for generating semantic explanations of the decision-making processes in deep neural networks, having applications in both model improvement and understanding. It is vital that the explanation is accurate, or faithful, to the model, yet we identify several limitations of prior faithfulness metrics that inhibit an accurate evaluation; most notably, prior metrics involve only the set of concepts present, ignoring how they may be spatially distributed. We address these limitations with Surrogate Faithfulness (SF), an evaluation method that introduces a spatially-aware surrogate and two novel faithfulness metrics. Using SF, we produce Optimally Faithful (OF) explanations, where concepts are found that maximize faithfulness. Our experiments show that (1) adding spatial-awareness to prior U-CBEMs increases faithfulness in all cases; (2) OF produces significantly more faithful explanations than prior U-CBEMs (30% or higher improvement in error); (3) OF's learned concepts generalize well to out-of-domain data and are more robust to adversarial examples, where prior U-CBEMs struggle. 

**Abstract (ZH)**: 基于概念的后验无监督解释方法（U-CBEMs）的空间 Awareness 评估及其优化可靠解释 

---
# Progressive Rock Music Classification 

**Title (ZH)**: 渐进摇滚音乐分类 

**Authors**: Arpan Nagar, Joseph Bensabat, Jokent Gaza, Moinak Dey  

**Link**: [PDF](https://arxiv.org/pdf/2504.10821)  

**Abstract**: This study investigates the classification of progressive rock music, a genre characterized by complex compositions and diverse instrumentation, distinct from other musical styles. Addressing this Music Information Retrieval (MIR) task, we extracted comprehensive audio features, including spectrograms, Mel-Frequency Cepstral Coefficients (MFCCs), chromagrams, and beat positions from song snippets using the Librosa library. A winner-take-all voting strategy was employed to aggregate snippet-level predictions into final song classifications. We conducted a comparative analysis of various machine learning techniques. Ensemble methods, encompassing Bagging (Random Forest, ExtraTrees, Bagging Classifier) and Boosting (XGBoost, Gradient Boosting), were explored, utilizing Principal Component Analysis (PCA) for dimensionality reduction to manage computational constraints with high-dimensional feature sets. Additionally, deep learning approaches were investigated, including the development of custom 1D Convolutional Neural Network (1D CNN) architectures (named "Zuck" and "Satya") featuring specific layer configurations, normalization, and activation functions. Furthermore, we fine-tuned a state-of-the-art Audio Spectrogram Transformer (AST) model, leveraging its attention-based mechanisms for audio classification. Performance evaluation on validation and test sets revealed varying effectiveness across models, with ensemble methods like Extra Trees achieving test accuracies up to 76.38%. This research provides insights into the application and relative performance of diverse machine learning paradigms for the nuanced task of progressive rock genre classification. 

**Abstract (ZH)**: 本研究探究了进步摇滚音乐的分类，该音乐风格以复杂的编排和多样的乐器配置为特点，与其他音乐风格明显不同。针对这一音乐信息检索（MIR）任务，我们从歌曲片段中使用Librosa库提取了包括频谱图、梅尔频率倒谱系数（MFCC）、chromagram和节拍位置在内的综合音频特征。采用了一票当选的投票策略，将片段级别的预测聚合为最终的歌曲分类。我们对比分析了多种机器学习技术。探索了包涵随机森林、额外树木、袋装分类器的自助集成方法，以及XGBoost、梯度提升等提升方法，并利用主成分分析（PCA）进行降维以管理高维特征集带来的计算约束。此外，我们还研究了深度学习方法，包括开发了特定层配置、归一化和激活函数的定制一维卷积神经网络（1D CNN，分别称为“Zuck”和“Satya”）。进一步地，我们对最新的音频光谱变换器（AST）模型进行了微调，利用其基于注意力机制的技术进行音频分类。对验证集和测试集的性能评估结果显示，不同模型的效果各异，例如额外树木集成方法在测试集上的准确率可达到76.38%。本研究为多种机器学习范式在精细的进步摇滚音乐类型分类任务中的应用和相对性能提供了见解。 

---
# FHBench: Towards Efficient and Personalized Federated Learning for Multimodal Healthcare 

**Title (ZH)**: FHBench: 向高效的个性化多模态医疗联邦学习迈进 

**Authors**: Penghao Wang, Qian Chen, Teng Zhang, Yingwei Zhang, Wang Lu, Yiqiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.10817)  

**Abstract**: Federated Learning (FL) has emerged as an effective solution for multi-institutional collaborations without sharing patient data, offering a range of methods tailored for diverse applications. However, real-world medical datasets are often multimodal, and computational resources are limited, posing significant challenges for existing FL approaches. Recognizing these limitations, we developed the Federated Healthcare Benchmark(FHBench), a benchmark specifically designed from datasets derived from real-world healthcare applications. FHBench encompasses critical diagnostic tasks across domains such as the nervous, cardiovascular, and respiratory systems and general pathology, providing comprehensive support for multimodal healthcare evaluations and filling a significant gap in existing benchmarks. Building on FHBench, we introduced Efficient Personalized Federated Learning with Adaptive LoRA(EPFL), a personalized FL framework that demonstrates superior efficiency and effectiveness across various healthcare modalities. Our results highlight the robustness of FHBench as a benchmarking tool and the potential of EPFL as an innovative approach to advancing healthcare-focused FL, addressing key limitations of existing methods. 

**Abstract (ZH)**: 联邦学习(Federated Learning, FL)已成为多机构合作而不共享患者数据的有效解决方案，提供了多种针对不同应用的定制方法。然而，现实世界的医疗数据集往往是多模态的，并且计算资源有限，给现有的FL方法带来了重大挑战。针对这些限制，我们开发了联邦医疗保健基准(Federated Healthcare Benchmark, FHBench)，这是一个专为来自实际医疗保健应用的数据集设计的基准。FHBench涵盖了神经系统、心血管系统、呼吸系统以及普通病理等领域的关键诊断任务，提供了全面的支持以进行多模态医疗评估，并填补了现有基准中的一个重大空白。基于FHBench，我们引入了高效个性化联邦学习与自适应LoRA(EPFL)，这是一种跨各种医疗模态具有优越效率和效果的个性化FL框架。我们的结果强调了FHBench作为基准测试工具的稳健性，并突显了EPFL作为推进以医疗保健为重点的FL的创新方法的潜力，解决了现有方法的关键限制。 

---
# Visual Language Models show widespread visual deficits on neuropsychological tests 

**Title (ZH)**: 视觉语言模型在神经心理测试中普遍表现出视觉缺陷 

**Authors**: Gene Tangtartharakul, Katherine R. Storrs  

**Link**: [PDF](https://arxiv.org/pdf/2504.10786)  

**Abstract**: Visual Language Models (VLMs) show remarkable performance in visual reasoning tasks, successfully tackling college-level challenges that require high-level understanding of images. However, some recent reports of VLMs struggling to reason about elemental visual concepts like orientation, position, continuity, and occlusion suggest a potential gulf between human and VLM vision. Here we use the toolkit of neuropsychology to systematically assess the capabilities of three state-of-the-art VLMs across visual domains. Using 51 tests drawn from six clinical and experimental batteries, we characterise the visual abilities of leading VLMs relative to normative performance in healthy adults. While the models excel in straightforward object recognition tasks, we find widespread deficits in low- and mid-level visual abilities that would be considered clinically significant in humans. These selective deficits, profiled through validated test batteries, suggest that an artificial system can achieve complex object recognition without developing foundational visual concepts that in humans require no explicit training. 

**Abstract (ZH)**: 视觉语言模型在视觉推理任务中展现出显著性能，成功应对了涉及高级图像理解的大学级挑战。然而，近期关于视觉语言模型在处理诸如方向、位置、连续性和遮挡等基本视觉概念方面的困难报告表明，人类与视觉语言模型的视觉之间可能存在巨大的差距。我们使用神经心理学工具，系统地评估了三种最先进的视觉语言模型在视觉领域的能力。通过来自六个临床和实验量表的51项测试，我们量化了领先视觉语言模型在视觉能力上的表现，对照健康成年人的正常表现进行比较。虽然这些模型在简单的物体识别任务上表现出色，但在低级和中级视觉能力方面我们发现了广泛存在的缺陷，这些缺陷在人类中被认为是临床显著的。这些选择性的缺陷，通过有效的测试量表进行刻画，表明一个人工系统可以在没有发展出人类在没有任何明确训练需求的情况下所需的基本视觉概念的情况下，实现复杂的物体识别能力。 

---
# Neural Network Emulation of the Classical Limit in Quantum Systems via Learned Observable Mappings 

**Title (ZH)**: 通过学习可观测量映射在量子系统中模拟经典极限的神经网络 

**Authors**: Kamran Majid  

**Link**: [PDF](https://arxiv.org/pdf/2504.10781)  

**Abstract**: The classical limit of quantum mechanics, formally investigated through frameworks like strict deformation quantization, remains a profound area of inquiry in the philosophy of physics. This paper explores a computational approach employing a neural network to emulate the emergence of classical behavior from the quantum harmonic oscillator as Planck's constant $\hbar$ approaches zero. We develop and train a neural network architecture to learn the mapping from initial expectation values and $\hbar$ to the time evolution of the expectation value of position. By analyzing the network's predictions across different regimes of hbar, we aim to provide computational insights into the nature of the quantum-classical transition. This work demonstrates the potential of machine learning as a complementary tool for exploring foundational questions in quantum mechanics and its classical limit. 

**Abstract (ZH)**: 量子力学的经典极限，通过严格的变形量ization等框架形式研究，仍然是物理学哲学中的一个深刻探究领域。本文探讨了一种采用神经网络的计算方法，模拟普朗克常数$\hbar$趋近于零时量子简谐振子的经典行为 Emergence of Classical Behavior from the Quantum Harmonic Oscillator as $\hbar$ Approaches Zero via a Neural Network：通过神经网络探讨量子简谐振子的经典行为随着$\hbar$趋近于零的涌现，旨在提供量子经典过渡性质的计算见解。本文展示了机器学习作为一种探索量子力学及其经典极限基础问题的补充工具的潜在价值。 

---
# Epistemic Uncertainty-aware Recommendation Systems via Bayesian Deep Ensemble Learning 

**Title (ZH)**: 基于贝叶斯深度集成学习的 Epistemic 不确定性感知推荐系统 

**Authors**: Radin Cheraghi, Amir Mohammad Mahfoozi, Sepehr Zolfaghari, Mohammadshayan Shabani, Maryam Ramezani, Hamid R. Rabiee  

**Link**: [PDF](https://arxiv.org/pdf/2504.10753)  

**Abstract**: Recommending items to users has long been a fundamental task, and studies have tried to improve it ever since. Most well-known models commonly employ representation learning to map users and items into a unified embedding space for matching assessment. These approaches have primary limitations, especially when dealing with explicit feedback and sparse data contexts. Two primary limitations are their proneness to overfitting and failure to incorporate epistemic uncertainty in predictions. To address these problems, we propose a novel Bayesian Deep Ensemble Collaborative Filtering method named BDECF. To improve model generalization and quality, we utilize Bayesian Neural Networks, which incorporate uncertainty within their weight parameters. In addition, we introduce a new interpretable non-linear matching approach for the user and item embeddings, leveraging the advantages of the attention mechanism. Furthermore, we endorse the implementation of an ensemble-based supermodel to generate more robust and reliable predictions, resulting in a more complete model. Empirical evaluation through extensive experiments and ablation studies across a range of publicly accessible real-world datasets with differing sparsity characteristics confirms our proposed method's effectiveness and the importance of its components. 

**Abstract (ZH)**: 推荐用户项目一直是基本任务，研究者们一直在努力改进这一任务。绝大多数知名模型通常采用表示学习将用户和项目映射到统一的嵌入空间以进行匹配评估。这些方法的主要局限性，在处理显式反馈和稀疏数据上下文时尤为突出。两个主要局限性包括容易过拟合和无法在预测中整合先验不确定性。为解决这些问题，我们提出了一种新的贝叶斯深度集成协作过滤方法，命名为BDECF。为提高模型的泛化能力和质量，我们采用了贝叶斯神经网络，这种方法在其权重参数中整合了不确定性。此外，我们引入了一种新的可解释的非线性匹配方法，利用注意机制的优势，为用户和项目嵌入提供匹配方法。同时，我们倡导使用基于集成的超模型来生成更稳健和可靠的预测，从而构建一个更为完整的方法。通过广泛实验和跨多种公共真实世界稀疏特性各异的数据集的消融研究，实证评估证实了我们所提方法的有效性及其各个组件的重要性。 

---
# Hearing Anywhere in Any Environment 

**Title (ZH)**: 在任意环境下的 anytime  hearing 

**Authors**: Xiulong Liu, Anurag Kumar, Paul Calamia, Sebastia V. Amengual, Calvin Murdock, Ishwarya Ananthabhotla, Philip Robinson, Eli Shlizerman, Vamsi Krishna Ithapu, Ruohan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.10746)  

**Abstract**: In mixed reality applications, a realistic acoustic experience in spatial environments is as crucial as the visual experience for achieving true immersion. Despite recent advances in neural approaches for Room Impulse Response (RIR) estimation, most existing methods are limited to the single environment on which they are trained, lacking the ability to generalize to new rooms with different geometries and surface materials. We aim to develop a unified model capable of reconstructing the spatial acoustic experience of any environment with minimum additional measurements. To this end, we present xRIR, a framework for cross-room RIR prediction. The core of our generalizable approach lies in combining a geometric feature extractor, which captures spatial context from panorama depth images, with a RIR encoder that extracts detailed acoustic features from only a few reference RIR samples. To evaluate our method, we introduce ACOUSTICROOMS, a new dataset featuring high-fidelity simulation of over 300,000 RIRs from 260 rooms. Experiments show that our method strongly outperforms a series of baselines. Furthermore, we successfully perform sim-to-real transfer by evaluating our model on four real-world environments, demonstrating the generalizability of our approach and the realism of our dataset. 

**Abstract (ZH)**: 在混合现实应用中，空间环境中的真实声学体验与视觉体验一样至关重要，对于实现真正的沉浸感至关重要。尽管在基于神经网络的房间冲激响应（RIR）估计方面取得了近期进展，但大多数现有方法仅限于它们所训练的单一环境，缺乏在不同几何结构和表面材料的新房间中泛化的能力。我们旨在开发一种统一模型，能够在最少的额外测量下重构任何环境的空间声学体验。为此，我们提出了xRIR，一种跨房间RIR预测框架。我们泛化方法的核心在于结合一个几何特征提取器，该提取器从全景深度图像中捕获空间上下文，与一个仅从少量参考RIR样本中提取详细声学特征的RIR编码器相结合。为了评估我们的方法，我们引入了ACOUSTICROOMS新数据集，该数据集包含来自260个房间的超过300,000个高保真模拟的RIR。实验结果显示，我们的方法显著优于一系列基线方法。此外，我们成功地通过在四个真实世界环境中评估我们的模型展示了方法的泛化能力和数据集的真实感。 

---
# Frozen Layers: Memory-efficient Many-fidelity Hyperparameter Optimization 

**Title (ZH)**: 冻结层：高效多保真度超参数优化 

**Authors**: Timur Carstensen, Neeratyoy Mallik, Frank Hutter, Martin Rapp  

**Link**: [PDF](https://arxiv.org/pdf/2504.10735)  

**Abstract**: As model sizes grow, finding efficient and cost-effective hyperparameter optimization (HPO) methods becomes increasingly crucial for deep learning pipelines. While multi-fidelity HPO (MF-HPO) trades off computational resources required for DL training with lower fidelity estimations, existing fidelity sources often fail under lower compute and memory constraints. We propose a novel fidelity source: the number of layers that are trained or frozen during training. For deep networks, this approach offers significant compute and memory savings while preserving rank correlations between hyperparameters at low fidelities compared to full model training. We demonstrate this in our empirical evaluation across ResNets and Transformers and additionally analyze the utility of frozen layers as a fidelity in using GPU resources as a fidelity in HPO, and for a combined MF-HPO with other fidelity sources. This contribution opens new applications for MF-HPO with hardware resources as a fidelity and creates opportunities for improved algorithms navigating joint fidelity spaces. 

**Abstract (ZH)**: 随着模型规模的增长，寻找高效的成本优化超参数优化（HPO）方法对于深度学习流水线变得越来越重要。尽管多保真度HPO（MF-HPO）在降低计算资源消耗的同时提供较低保真度的估计，但现有的保真度源在较低的计算和内存约束下常常失效。我们提出了一种新的保真度源：在训练过程中被训练或冻结的层的数量。对于深层网络，这种方法在保留低保真度下超参数之间的排序相关性的同时，提供了显著的计算和内存节省。我们在ResNets和Transformers的实证评估中展示了这一点，并进一步分析了冻结层作为保真度在使用GPU资源进行HPO以及与其他保真度源结合的多保真度HPO中的效用。这一贡献为将硬件资源作为保真度的多保真度HPO开辟了新的应用，并为在联合保真度空间中寻找改进算法提供了机会。 

---
# Optimizing Data Distribution and Kernel Performance for Efficient Training of Chemistry Foundation Models: A Case Study with MACE 

**Title (ZH)**: 化学基础模型高效训练中的数据分布优化与核函数性能提升：MACE案例研究 

**Authors**: Jesun Firoz, Franco Pellegrini, Mario Geiger, Darren Hsu, Jenna A. Bilbrey, Han-Yi Chou, Maximilian Stadler, Markus Hoehnerbach, Tingyu Wang, Dejun Lin, Emine Kucukbenli, Henry W. Sprueill, Ilyes Batatia, Sotiris S. Xantheas, MalSoon Lee, Chris Mundy, Gabor Csanyi, Justin S. Smith, Ponnuswamy Sadayappan, Sutanay Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.10700)  

**Abstract**: Chemistry Foundation Models (CFMs) that leverage Graph Neural Networks (GNNs) operating on 3D molecular graph structures are becoming indispensable tools for computational chemists and materials scientists. These models facilitate the understanding of matter and the discovery of new molecules and materials. In contrast to GNNs operating on a large homogeneous graphs, GNNs used by CFMs process a large number of geometric graphs of varying sizes, requiring different optimization strategies than those developed for large homogeneous GNNs. This paper presents optimizations for two critical phases of CFM training: data distribution and model training, targeting MACE - a state-of-the-art CFM. We address the challenge of load balancing in data distribution by formulating it as a multi-objective bin packing problem. We propose an iterative algorithm that provides a highly effective, fast, and practical solution, ensuring efficient data distribution. For the training phase, we identify symmetric tensor contraction as the key computational kernel in MACE and optimize this kernel to improve the overall performance. Our combined approach of balanced data distribution and kernel optimization significantly enhances the training process of MACE. Experimental results demonstrate a substantial speedup, reducing per-epoch execution time for training from 12 to 2 minutes on 740 GPUs with a 2.6M sample dataset. 

**Abstract (ZH)**: 基于图神经网络的化学基础模型中的优化研究：以MACE为例 

---
# NTIRE 2025 Challenge on Cross-Domain Few-Shot Object Detection: Methods and Results 

**Title (ZH)**: NTIRE 2025挑战赛：跨域少样本对象检测——方法与结果 

**Authors**: Yuqian Fu, Xingyu Qiu, Bin Ren, Yanwei Fu, Radu Timofte, Nicu Sebe, Ming-Hsuan Yang, Luc Van Gool, Kaijin Zhang, Qingpeng Nong, Xiugang Dong, Hong Gao, Xiangsheng Zhou, Jiancheng Pan, Yanxing Liu, Xiao He, Jiahao Li, Yuze Sun, Xiaomeng Huang, Zhenyu Zhang, Ran Ma, Yuhan Liu, Zijian Zhuang, Shuai Yi, Yixiong Zou, Lingyi Hong, Mingxi Chen, Runze Li, Xingdong Sheng, Wenqiang Zhang, Weisen Chen, Yongxin Yan, Xinguo Chen, Yuanjie Shao, Zhengrong Zuo, Nong Sang, Hao Wu, Haoran Sun, Shuming Hu, Yan Zhang, Zhiguang Shi, Yu Zhang, Chao Chen, Tao Wang, Da Feng, Linhai Zhuo, Ziming Lin, Yali Huang, Jie Me, Yiming Yang, Mi Guo, Mingyuan Jiu, Mingliang Xu, Maomao Xiong, Qunshu Zhang, Xinyu Cao, Yuqing Yang, Dianmo Sheng, Xuanpu Zhao, Zhiyu Li, Xuyang Ding, Wenqian Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10685)  

**Abstract**: Cross-Domain Few-Shot Object Detection (CD-FSOD) poses significant challenges to existing object detection and few-shot detection models when applied across domains. In conjunction with NTIRE 2025, we organized the 1st CD-FSOD Challenge, aiming to advance the performance of current object detectors on entirely novel target domains with only limited labeled data. The challenge attracted 152 registered participants, received submissions from 42 teams, and concluded with 13 teams making valid final submissions. Participants approached the task from diverse perspectives, proposing novel models that achieved new state-of-the-art (SOTA) results under both open-source and closed-source settings. In this report, we present an overview of the 1st NTIRE 2025 CD-FSOD Challenge, highlighting the proposed solutions and summarizing the results submitted by the participants. 

**Abstract (ZH)**: 跨域少样本对象检测（CD-FSOD）在不同领域应用时对现有对象检测和少样本检测模型提出了重大挑战。为促进对象检测器在全新目标领域上的性能提升，仅凭有限标注数据，我们与NTIRE 2025合作组织了第1届CD-FSOD挑战赛。该挑战赛吸引了152名注册参赛者，共收到42支队伍的提交，并有13支队伍提交了有效的最终结果。参赛者从多个角度出发，提出了新的模型，在开源和闭源环境下均取得了新的最先进（SOTA）成果。本文报告了第1届NTIRE 2025 CD-FSOD挑战赛的概述，强调了提出的方法并总结了参赛者的提交结果。 

---
# Keyword Extraction, and Aspect Classification in Sinhala, English, and Code-Mixed Content 

**Title (ZH)**: Sinhala、English及其代码混合内容中的关键信息提取与aspect分类 

**Authors**: F.A. Rizvi, T. Navojith, A.M.N.H. Adhikari, W.P.U. Senevirathna, Dharshana Kasthurirathna, Lakmini Abeywardhana  

**Link**: [PDF](https://arxiv.org/pdf/2504.10679)  

**Abstract**: Brand reputation in the banking sector is maintained through insightful analysis of customer opinion on code-mixed and multilingual content. Conventional NLP models misclassify or ignore code-mixed text, when mix with low resource languages such as Sinhala-English and fail to capture domain-specific knowledge. This study introduces a hybrid NLP method to improve keyword extraction, content filtering, and aspect-based classification of banking content. Keyword extraction in English is performed with a hybrid approach comprising a fine-tuned SpaCy NER model, FinBERT-based KeyBERT embeddings, YAKE, and EmbedRank, which results in a combined accuracy of 91.2%. Code-mixed and Sinhala keywords are extracted using a fine-tuned XLM-RoBERTa model integrated with a domain-specific Sinhala financial vocabulary, and it results in an accuracy of 87.4%. To ensure data quality, irrelevant comment filtering was performed using several models, with the BERT-base-uncased model achieving 85.2% for English and XLM-RoBERTa 88.1% for Sinhala, which was better than GPT-4o, SVM, and keyword-based filtering. Aspect classification followed the same pattern, with the BERT-base-uncased model achieving 87.4% for English and XLM-RoBERTa 85.9% for Sinhala, both exceeding GPT-4 and keyword-based approaches. These findings confirm that fine-tuned transformer models outperform traditional methods in multilingual financial text analysis. The present framework offers an accurate and scalable solution for brand reputation monitoring in code-mixed and low-resource banking environments. 

**Abstract (ZH)**: 银行领域的品牌声誉通过深入分析混合语言和多语言内容的客户意见来维护。传统的NLP模型在低资源语言（如僧伽罗语-英语混合文本）中会出现误分类或忽略，并且难以捕捉领域特定知识。本研究引入了一种混合NLP方法，以改进银行内容的关键词提取、内容过滤和方面分类。关键词提取在英语中采用了一种混合方法，结合了微调的SpaCy NER模型、FinBERT基的KeyBERT嵌入、YAKE和EmbedRank，最终准确率为91.2%。僧伽罗语和混合语言关键词采用集成域特定僧伽罗语金融词汇的微调XLM-RoBERTa模型进行提取，准确率为87.4%。为保证数据质量，不相关的评论过滤使用了多种模型，其中BERT-base-uncased模型在英语中的准确率为85.2%，XLM-RoBERTa模型在僧伽罗语中的准确率为88.1%，优于GPT-4o、SVM和基于关键词的过滤。方面分类也遵循相同的模式，BERT-base-uncased模型在英语中的准确率为87.4%，XLM-RoBERTa模型在僧伽罗语中的准确率为85.9%，均超过了GPT-4和基于关键词的方法。这些发现证实，微调的变换器模型在多语言金融文本分析中优于传统方法。当前框架为混合语言和低资源银行环境中的品牌声誉监控提供了准确且可扩展的解决方案。 

---
# Achieving Optimal Tissue Repair Through MARL with Reward Shaping and Curriculum Learning 

**Title (ZH)**: 通过奖励塑造和分级学习实现最优组织修复的MARL方法 

**Authors**: Muhammad Al-Zafar Khan, Jamal Al-Karaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.10677)  

**Abstract**: In this paper, we present a multi-agent reinforcement learning (MARL) framework for optimizing tissue repair processes using engineered biological agents. Our approach integrates: (1) stochastic reaction-diffusion systems modeling molecular signaling, (2) neural-like electrochemical communication with Hebbian plasticity, and (3) a biologically informed reward function combining chemical gradient tracking, neural synchronization, and robust penalties. A curriculum learning scheme guides the agent through progressively complex repair scenarios. In silico experiments demonstrate emergent repair strategies, including dynamic secretion control and spatial coordination. 

**Abstract (ZH)**: 本研究提出了一种多智能体强化学习（MARL）框架，用于利用工程生物代理人优化组织修复过程。该方法集成如下内容：（1）随机反应-扩散系统建模分子信号传导，（2）具有 Hebbian 可塑性的神经似信号化学通信，以及（3）一种基于化学梯度跟踪、神经同步和鲁棒惩罚的生物启发式奖励函数。通过阶梯式学习方案引导智能体逐步通过复杂度渐增的修复场景。计算机模拟实验展示了涌现的修复策略，包括动态分泌控制和空间协调。 

---
# Characterizing Knowledge Manipulation in a Russian Wikipedia Fork 

**Title (ZH)**: characterizing知识操纵在俄罗斯维基百科分支中 

**Authors**: Mykola Trokhymovych, Oleksandr Kosovan, Nathan Forrester, Pablo Aragón, Diego Saez-Trumper, Ricardo Baeza-Yates  

**Link**: [PDF](https://arxiv.org/pdf/2504.10663)  

**Abstract**: Wikipedia is powered by MediaWiki, a free and open-source software that is also the infrastructure for many other wiki-based online encyclopedias. These include the recently launched website Ruwiki, which has copied and modified the original Russian Wikipedia content to conform to Russian law. To identify practices and narratives that could be associated with different forms of knowledge manipulation, this article presents an in-depth analysis of this Russian Wikipedia fork. We propose a methodology to characterize the main changes with respect to the original version. The foundation of this study is a comprehensive comparative analysis of more than 1.9M articles from Russian Wikipedia and its fork. Using meta-information and geographical, temporal, categorical, and textual features, we explore the changes made by Ruwiki editors. Furthermore, we present a classification of the main topics of knowledge manipulation in this fork, including a numerical estimation of their scope. This research not only sheds light on significant changes within Ruwiki, but also provides a methodology that could be applied to analyze other Wikipedia forks and similar collaborative projects. 

**Abstract (ZH)**: Wikipedia由MediaWiki驱动，这是一个免费开源的软件，也是许多其他基于维基的在线百科全书的基础。这包括最近推出的Ruwiki网站，该网站复制并修改了原始的俄罗斯维基百科内容以符合俄罗斯法律规定。为了识别与不同形式的知识操纵相关的行为和叙事，本文对这个俄罗斯维基百科分支进行了深入分析。我们提出了一种方法来描述与原始版本相比的主要变化。本研究的基础是对俄罗斯维基百科及其分支超过190万篇文章进行了全面比较分析。借助元信息和地理、时间、类别以及文本特征，我们探讨了Ruwiki编辑者所做的更改。此外，我们对该分支中的主要知识操纵主题进行了分类，并对其范围进行了量化评估。这项研究不仅揭示了Ruwiki中的重要变化，还提供了一种可以应用于分析其他维基分支和类似协作项目的分析方法。 

---
# MatterTune: An Integrated, User-Friendly Platform for Fine-Tuning Atomistic Foundation Models to Accelerate Materials Simulation and Discovery 

**Title (ZH)**: MatterTune: 一个集成的、用户友好的平台，用于优化原子级基础模型以加速材料模拟与发现 

**Authors**: Lingyu Kong, Nima Shoghi, Guoxiang Hu, Pan Li, Victor Fung  

**Link**: [PDF](https://arxiv.org/pdf/2504.10655)  

**Abstract**: Geometric machine learning models such as graph neural networks have achieved remarkable success in recent years in chemical and materials science research for applications such as high-throughput virtual screening and atomistic simulations. The success of these models can be attributed to their ability to effectively learn latent representations of atomic structures directly from the training data. Conversely, this also results in high data requirements for these models, hindering their application to problems which are data sparse which are common in this domain. To address this limitation, there is a growing development in the area of pre-trained machine learning models which have learned general, fundamental, geometric relationships in atomistic data, and which can then be fine-tuned to much smaller application-specific datasets. In particular, models which are pre-trained on diverse, large-scale atomistic datasets have shown impressive generalizability and flexibility to downstream applications, and are increasingly referred to as atomistic foundation models. To leverage the untapped potential of these foundation models, we introduce MatterTune, a modular and extensible framework that provides advanced fine-tuning capabilities and seamless integration of atomistic foundation models into downstream materials informatics and simulation workflows, thereby lowering the barriers to adoption and facilitating diverse applications in materials science. In its current state, MatterTune supports a number of state-of-the-art foundation models such as ORB, MatterSim, JMP, and EquformerV2, and hosts a wide range of features including a modular and flexible design, distributed and customizable fine-tuning, broad support for downstream informatics tasks, and more. 

**Abstract (ZH)**: 几何机器学习模型如图神经网络近年来在化学和材料科学研究中取得了显著成功，特别是在高通量虚拟筛选和原子尺度模拟应用方面。这些模型的成功归因于它们能够直接从训练数据中有效地学习原子结构的潜在表示。相反，这也导致这些模型需要高数据需求，从而阻碍了它们在数据稀疏问题中的应用，而这类问题在该领域非常常见。为了解决这一局限性，预训练机器学习模型的发展日益增长，这些模型已在原子尺度数据中学到了一般性和基础性的几何关系，并可进一步微调以适应较小的应用特定数据集。特别是，在多样化大规模原子尺度数据集上预训练的模型展现出了强大的泛化能力和下游应用的灵活性，并逐渐被称为原子尺度基础模型。为了充分利用这些基础模型的潜力，我们引入了MatterTune，这是一种模块化和可扩展的框架，提供了高级微调功能，并无缝整合原子尺度基础模型到下游材料informatics和模拟工作流中，从而降低了采用门槛并促进了材料科学中各种应用的实现。目前，MatterTune 支持包括 ORB、MatterSim、JMP 和 EquformerV2 在内的多种最先进的基础模型，并提供了包括模块化和灵活设计、分布式和可定制的微调、对下游informatics任务的广泛支持等一系列功能。 

---
# Will AI shape the way we speak? The emerging sociolinguistic influence of synthetic voices 

**Title (ZH)**: AI是否会塑造我们的语言方式？合成语音的新兴社会语言学影响。 

**Authors**: Éva Székely, Jūra Miniota, Míša, Hejná  

**Link**: [PDF](https://arxiv.org/pdf/2504.10650)  

**Abstract**: The growing prevalence of conversational voice interfaces, powered by developments in both speech and language technologies, raises important questions about their influence on human communication. While written communication can signal identity through lexical and stylistic choices, voice-based interactions inherently amplify socioindexical elements - such as accent, intonation, and speech style - which more prominently convey social identity and group affiliation. There is evidence that even passive media such as television is likely to influence the audience's linguistic patterns. Unlike passive media, conversational AI is interactive, creating a more immersive and reciprocal dynamic that holds a greater potential to impact how individuals speak in everyday interactions. Such heightened influence can be expected to arise from phenomena such as acoustic-prosodic entrainment and linguistic accommodation, which occur naturally during interaction and enable users to adapt their speech patterns in response to the system. While this phenomenon is still emerging, its potential societal impact could provide organisations, movements, and brands with a subtle yet powerful avenue for shaping and controlling public perception and social identity. We argue that the socioindexical influence of AI-generated speech warrants attention and should become a focus of interdisciplinary research, leveraging new and existing methodologies and technologies to better understand its implications. 

**Abstract (ZH)**: 基于语音技术发展的对话式语音接口日益普及，这对人类交流产生了重要影响。虽然书面交流可以通过词汇和风格选择来信号身份，但基于语音的交互会更显著地放大社会指数性特征，如口音、语调和言语风格，这些特征更直接地传达了社会身份和群体归属感。有证据表明，即使是电视这样的被动媒体也会影响观众的语言模式。与被动媒体不同，对话式AI具有互动性，能够创建一种更沉浸和互动的动力学，具有更大的潜力影响个体在日常交流中的言语方式。这种增强的影响可以预期源于诸如声学-语调同步和语言调整等自然现象，这些现象在互动过程中会发生，使用户能够根据系统的响应调整自己的言语模式。尽管这一现象仍处于初期阶段，但其潜在的社会影响可能会为组织、运动和品牌提供一种微妙而有力的途径，以塑造和控制公共认知和社会身份。我们主张，应关注AI生成语音的社会指数性影响，并将其作为跨学科研究的重点，利用新旧方法和技术来更好地理解其影响。 

---
# Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling 

**Title (ZH)**: 能量匹配：统一流匹配和能量基于模型的生成建模 

**Authors**: Michal Balcerak, Tamaz Amiranashvili, Suprosanna Shit, Antonio Terpin, Sebastian Kaltenbach, Petros Koumoutsakos, Bjoern Menze  

**Link**: [PDF](https://arxiv.org/pdf/2504.10612)  

**Abstract**: Generative models often map noise to data by matching flows or scores, but these approaches become cumbersome for incorporating partial observations or additional priors. Inspired by recent advances in Wasserstein gradient flows, we propose Energy Matching, a framework that unifies flow-based approaches with the flexibility of energy-based models (EBMs). Far from the data manifold, samples move along curl-free, optimal transport paths from noise to data. As they approach the data manifold, an entropic energy term guides the system into a Boltzmann equilibrium distribution, explicitly capturing the underlying likelihood structure of the data. We parameterize this dynamic with a single time-independent scalar field, which serves as both a powerful generator and a flexible prior for effective regularization of inverse problems. Our method substantially outperforms existing EBMs on CIFAR-10 generation (FID 3.97 compared to 8.61), while retaining the simulation-free training of transport-based approaches away from the data manifold. Additionally, we exploit the flexibility of our method and introduce an interaction energy for diverse mode exploration. Our approach focuses on learning a static scalar potential energy -- without time conditioning, auxiliary generators, or additional networks -- marking a significant departure from recent EBM methods. We believe this simplified framework significantly advances EBM capabilities and paves the way for their broader adoption in generative modeling across diverse domains. 

**Abstract (ZH)**: 基于能量匹配的生成模型 

---
# Self-Controlled Dynamic Expansion Model for Continual Learning 

**Title (ZH)**: 自我控制动态扩展模型 for 连续学习 

**Authors**: Runqing Wu, Fei Ye, Rongyao Hu, Guoxi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10561)  

**Abstract**: Continual Learning (CL) epitomizes an advanced training paradigm wherein prior data samples remain inaccessible during the acquisition of new tasks. Numerous investigations have delved into leveraging a pre-trained Vision Transformer (ViT) to enhance model efficacy in continual learning. Nonetheless, these approaches typically utilize a singular, static backbone, which inadequately adapts to novel tasks, particularly when engaging with diverse data domains, due to a substantial number of inactive parameters. This paper addresses this limitation by introducing an innovative Self-Controlled Dynamic Expansion Model (SCDEM), which orchestrates multiple distinct trainable pre-trained ViT backbones to furnish diverse and semantically enriched representations. Specifically, by employing the multi-backbone architecture as a shared module, the proposed SCDEM dynamically generates a new expert with minimal parameters to accommodate a new task. A novel Collaborative Optimization Mechanism (COM) is introduced to synergistically optimize multiple backbones by harnessing prediction signals from historical experts, thereby facilitating new task learning without erasing previously acquired knowledge. Additionally, a novel Feature Distribution Consistency (FDC) approach is proposed to align semantic similarity between previously and currently learned representations through an optimal transport distance-based mechanism, effectively mitigating negative knowledge transfer effects. Furthermore, to alleviate over-regularization challenges, this paper presents a novel Dynamic Layer-Wise Feature Attention Mechanism (DLWFAM) to autonomously determine the penalization intensity on each trainable representation layer. An extensive series of experiments have been conducted to evaluate the proposed methodology's efficacy, with empirical results corroborating that the approach attains state-of-the-art performance. 

**Abstract (ZH)**: 持续学习（CL）体现在一个先进的训练 paradigm 中，其中先前的数据样本在学习新任务时保持不可访问。众多研究致力于利用预训练的 Vision Transformer（ViT）来增强在持续学习中的模型效果。然而，这些方法通常使用单一的静态主干，这在处理多样化的数据领域时难以适应新的任务，尤其是由于大量未激活的参数。本文通过引入一种创新的自我控制动态扩展模型（SCDEM），解决了这一局限性。该模型协调多个可训练的预训练 ViT 主干，提供多样化的语义丰富表示。具体而言，通过使用多主干架构作为共享模块，所提出的 SCDEM 动态生成一个新的专家，以最小的参数量适应新任务。作者引入了一种新颖的合作优化机制（COM），通过利用历史专家的预测信号协同优化多个主干，从而在不丢失之前获取的知识的情况下促进新任务的学习。此外，提出了一个新颖的特征分布一致性（FDC）方法，通过最优运输距离机制对齐已学习和当前学习的表示之间的语义相似性，有效减轻负面知识转移的影响。为进一步缓解过度正则化挑战，本文提出了一种新颖的动态逐层特征注意机制（DLWFAM），以自主确定每个可训练表示层的惩罚强度。进行了大量的实验评估所提方法的有效性，实验证明该方法达到了最先进的性能。 

---
# VAE-based Feature Disentanglement for Data Augmentation and Compression in Generalized GNSS Interference Classification 

**Title (ZH)**: 基于VAE的特征解耦在广义GNSS干扰分类中的数据增强与压缩 

**Authors**: Lucas Heublein, Simon Kocher, Tobias Feigl, Alexander Rügamer, Christopher Mutschler, Felix Ott  

**Link**: [PDF](https://arxiv.org/pdf/2504.10556)  

**Abstract**: Distributed learning and Edge AI necessitate efficient data processing, low-latency communication, decentralized model training, and stringent data privacy to facilitate real-time intelligence on edge devices while reducing dependency on centralized infrastructure and ensuring high model performance. In the context of global navigation satellite system (GNSS) applications, the primary objective is to accurately monitor and classify interferences that degrade system performance in distributed environments, thereby enhancing situational awareness. To achieve this, machine learning (ML) models can be deployed on low-resource devices, ensuring minimal communication latency and preserving data privacy. The key challenge is to compress ML models while maintaining high classification accuracy. In this paper, we propose variational autoencoders (VAEs) for disentanglement to extract essential latent features that enable accurate classification of interferences. We demonstrate that the disentanglement approach can be leveraged for both data compression and data augmentation by interpolating the lower-dimensional latent representations of signal power. To validate our approach, we evaluate three VAE variants - vanilla, factorized, and conditional generative - on four distinct datasets, including two collected in controlled indoor environments and two real-world highway datasets. Additionally, we conduct extensive hyperparameter searches to optimize performance. Our proposed VAE achieves a data compression rate ranging from 512 to 8,192 and achieves an accuracy up to 99.92%. 

**Abstract (ZH)**: 分布式学习与边缘AI需要高效的数据处理、低延迟通信、去中心化的模型训练以及严格的数据隐私保护，以在边缘设备上实现实时智能，减少对中心化基础设施的依赖并保证高模型性能。在全球导航卫星系统（GNSS）应用中，主要目标是准确监控和分类影响系统性能的干扰，从而增强态势感知。为此，可以在低资源设备上部署机器学习（ML）模型，确保最小化通信延迟并保护数据隐私。关键挑战是如何在保持高分类精度的同时压缩ML模型。在本文中，我们提出使用变分自编码器（VAEs）进行解耦，以提取能够准确分类干扰的关键潜在特征。我们证明了解耦方法可以用于数据压缩和数据增强，通过插值信号功率的低维潜在表示。为了验证我们的方法，我们在四个不同的数据集上评估了三种VAE变体——vanilla、因子分解和条件生成型，包括两个在受控室内环境收集的数据集和两个实际高速公路数据集。此外，我们还进行了广泛的超参数搜索以优化性能。我们提出的方法达到了512到8,192的数据压缩率，并且分类精度高达99.92%。 

---
# Beyond the Generative Learning Trilemma: Generative Model Assessment in Data Scarcity Domains 

**Title (ZH)**: 超越生成学习三难境地：在数据稀缺领域中的生成模型评估 

**Authors**: Marco Salmè, Lorenzo Tronchin, Rosa Sicilia, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10555)  

**Abstract**: Data scarcity remains a critical bottleneck impeding technological advancements across various domains, including but not limited to medicine and precision agriculture. To address this challenge, we explore the potential of Deep Generative Models (DGMs) in producing synthetic data that satisfies the Generative Learning Trilemma: fidelity, diversity, and sampling efficiency. However, recognizing that these criteria alone are insufficient for practical applications, we extend the trilemma to include utility, robustness, and privacy, factors crucial for ensuring the applicability of DGMs in real-world scenarios. Evaluating these metrics becomes particularly challenging in data-scarce environments, as DGMs traditionally rely on large datasets to perform optimally. This limitation is especially pronounced in domains like medicine and precision agriculture, where ensuring acceptable model performance under data constraints is vital. To address these challenges, we assess the Generative Learning Trilemma in data-scarcity settings using state-of-the-art evaluation metrics, comparing three prominent DGMs: Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Diffusion Models (DMs). Furthermore, we propose a comprehensive framework to assess utility, robustness, and privacy in synthetic data generated by DGMs. Our findings demonstrate varying strengths among DGMs, with each model exhibiting unique advantages based on the application context. This study broadens the scope of the Generative Learning Trilemma, aligning it with real-world demands and providing actionable guidance for selecting DGMs tailored to specific applications. 

**Abstract (ZH)**: 数据稀缺仍然是阻碍医学和精准农业等领域技术进步的关键瓶颈。为应对这一挑战，我们探索了深度生成模型（DGMs）在满足生成学习三难局面（忠实性、多样性和采样效率）方面生成合成数据的潜力。然而，认识到这些标准本身不足以满足实际应用需求，我们将三难局面扩展到包括效用、稳健性和隐私性等因素，这些因素对于确保DGMs在实际场景中的适用性至关重要。在数据稀缺环境中评估这些指标变得尤为挑战性，因为传统的DGMs需要大量数据才能发挥最佳性能。这种限制在医学和精准农业等领域尤为明显，这些领域在数据受限条件下保证模型性能的可接受性至关重要。为应对这些挑战，我们使用最先进的评估指标，在数据稀缺条件下评估生成学习三难局面，比较三种突出的DGMs：变分自编码器（VAEs）、生成对抗网络（GANs）和扩散模型（DMs）。此外，我们提出了一种全面框架来评估由DGMs生成的合成数据的效用、稳健性和隐私性。我们的研究结果表明，DGMs在性能上有不同的优势，每种模型根据应用场景具有独特的优点。本研究扩大了生成学习三难局面的适用范围，使其与实际需求相契合，并提供了选择适用于特定应用的DGMs的实际指导。 

---
# MiMu: Mitigating Multiple Shortcut Learning Behavior of Transformers 

**Title (ZH)**: MiMu: 减缓 transformers 的多重捷径学习行为 

**Authors**: Lili Zhao, Qi Liu, Wei Chen, Liyi Chen, Ruijun Sun, Min Hou, Yang Wang, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10551)  

**Abstract**: Empirical Risk Minimization (ERM) models often rely on spurious correlations between features and labels during the learning process, leading to shortcut learning behavior that undermines robustness generalization performance. Current research mainly targets identifying or mitigating a single shortcut; however, in real-world scenarios, cues within the data are diverse and unknown. In empirical studies, we reveal that the models rely to varying extents on different shortcuts. Compared to weak shortcuts, models depend more heavily on strong shortcuts, resulting in their poor generalization ability. To address these challenges, we propose MiMu, a novel method integrated with Transformer-based ERMs designed to Mitigate Multiple shortcut learning behavior, which incorporates self-calibration strategy and self-improvement strategy. In the source model, we preliminarily propose the self-calibration strategy to prevent the model from relying on shortcuts and make overconfident predictions. Then, we further design self-improvement strategy in target model to reduce the reliance on multiple shortcuts. The random mask strategy involves randomly masking partial attention positions to diversify the focus of target model other than concentrating on a fixed region. Meanwhile, the adaptive attention alignment module facilitates the alignment of attention weights to the calibrated source model, without the need for post-hoc attention maps or supervision. Finally, extensive experiments conducted on Natural Language Processing (NLP) and Computer Vision (CV) demonstrate the effectiveness of MiMu in improving robustness generalization abilities. 

**Abstract (ZH)**: Mitigating Multiple Shortcut Learning Behavior in Empirical Risk Minimization Models through Integrated Transformer-based Methods 

---
# Physics-Informed Neural Networks for Enhanced Interface Preservation in Lattice Boltzmann Multiphase Simulations 

**Title (ZH)**: 基于物理信息的神经网络在格子玻尔兹曼双相流模拟中增强界面保持方法 

**Authors**: Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10539)  

**Abstract**: This paper presents an improved approach for preserving sharp interfaces in multiphase Lattice Boltzmann Method (LBM) simulations using Physics-Informed Neural Networks (PINNs). Interface diffusion is a common challenge in multiphase LBM, leading to reduced accuracy in simulating phenomena where interfacial dynamics are critical. We propose a coupled PINN-LBM framework that maintains interface sharpness while preserving the physical accuracy of the simulation. Our approach is validated through droplet simulations, with quantitative metrics measuring interface width, maximum gradient, phase separation, effective interface width, and interface energy. The enhanced visualization techniques employed in this work clearly demonstrate the superior performance of PINN-LBM over standard LBM for multiphase simulations, particularly in maintaining well-defined interfaces throughout the simulation. We provide a comprehensive analysis of the results, showcasing how the neural network integration effectively counteracts numerical diffusion, while maintaining physical consistency with the underlying fluid dynamics. 

**Abstract (ZH)**: 本文提出了一种使用物理知情神经网络（PINNs）改善多相格子玻尔茲曼方法（LBM）仿真中保锐界面的方法。界面扩散是多相LBM中的常见挑战，会导致在界面动力学至关重要的现象模拟中降低准确性。我们提出了一种结合PINN-LBM框架，既能保持界面的锐利性又能保持仿真的物理准确性。我们的方法通过液滴仿真得到了验证，通过界面宽度、最大梯度、相分离、有效界面宽度和界面能量等定量指标进行评估。本工作中采用的增强可视化技术清楚地展示了PINN-LBM相较于标准LBM在多相仿真中优越的性能，尤其是在整个仿真过程中保持清晰定义的界面方面。我们对结果进行了全面分析，展示了神经网络集成如何有效地抵消数值扩散效应，同时保持与底层流体力学的一致性。 

---
# Integrating Emotion Distribution Networks and Textual Message Analysis for X User Emotional State Classification 

**Title (ZH)**: 整合情绪分布网络与文本消息分析进行X用户情绪状态分类 

**Authors**: Pardis Moradbeiki, Mohammad Ali Zare Chahooki  

**Link**: [PDF](https://arxiv.org/pdf/2504.10521)  

**Abstract**: As the popularity and reach of social networks continue to surge, a vast reservoir of opinions and sentiments across various subjects inundates these platforms. Among these, X social network (formerly Twitter) stands as a juggernaut, boasting approximately 420 million active users. Extracting users' emotional and mental states from their expressed opinions on social media has become a common pursuit. While past methodologies predominantly focused on the textual content of messages to analyze user sentiment, the interactive nature of these platforms suggests a deeper complexity. This study employs hybrid methodologies, integrating textual analysis, profile examination, follower analysis, and emotion dissemination patterns. Initially, user interactions are leveraged to refine emotion classification within messages, encompassing exchanges where users respond to each other. Introducing the concept of a communication tree, a model is extracted to map these interactions. Subsequently, users' bios and interests from this tree are juxtaposed with message text to enrich analysis. Finally, influential figures are identified among users' followers in the communication tree, categorized into different topics to gauge interests. The study highlights that traditional sentiment analysis methodologies, focusing solely on textual content, are inadequate in discerning sentiment towards significant events, notably the presidential election. Comparative analysis with conventional methods reveals a substantial improvement in accuracy with the incorporation of emotion distribution patterns and user profiles. The proposed approach yields a 12% increase in accuracy with emotion distribution patterns and a 15% increase when considering user profiles, underscoring its efficacy in capturing nuanced sentiment dynamics. 

**Abstract (ZH)**: 随着社交网络的 popularity 和 reach 不断增长，各种主题的意见和情感在这些平台上大量涌现。其中，X 社交网络（原 Twitter）作为巨头，拥有约 4.2 亿活跃用户。从用户在社交媒体上表达的意见中提取其情感和心理状态已成为一种常见追求。尽管以往的方法主要侧重于消息的文本内容来分析用户情感，但这些平台的互动性提示了更深层次的复杂性。本研究采用混合方法，结合文本分析、个人资料检查、关注者分析和情绪传播模式。首先，利用用户互动来细化消息中的情感分类，包括用户相互回应的情况。引入通信树的概念，提取模型以映射这些互动。随后，将通信树中的用户简介和兴趣与消息文本进行对比，以丰富分析。最后，在通信树中识别出用户的重要关注者，并按不同话题分类，以衡量兴趣。研究指出，专注于文本内容的传统情感分析方法在识别重要事件（尤其是总统选举）的情感时存在不足。与传统方法的比较分析表明，将情绪分布模式和用户资料纳入分析可显著提高准确性。提出的方法在情绪分布模式下准确率提高了 12%，在考虑用户资料时提高了 15%，突显了其在捕捉细微情感动态方面的有效性。 

---
# JEPA4Rec: Learning Effective Language Representations for Sequential Recommendation via Joint Embedding Predictive Architecture 

**Title (ZH)**: JEPA4Rec: 联合嵌入预测架构下有效语言表示学习的序列推荐方法 

**Authors**: Minh-Anh Nguyen, Dung D.Le  

**Link**: [PDF](https://arxiv.org/pdf/2504.10512)  

**Abstract**: Language representation learning has emerged as a promising approach for sequential recommendation, thanks to its ability to learn generalizable representations. However, despite its advantages, this approach still struggles with data sparsity and a limited understanding of common-sense user preferences. To address these limitations, we propose $\textbf{JEPA4Rec}$, a framework that combines $\textbf{J}$oint $\textbf{E}$mbedding $\textbf{P}$redictive $\textbf{A}$rchitecture with language modeling of item textual descriptions. JEPA4Rec captures semantically rich and transferable representations, improving recommendation performance and reducing reliance on large-scale pre-training data. Specifically, JEPA4Rec represents items as text sentences by flattening descriptive information such as $\textit{title, category}$, and other attributes. To encode these sentences, we employ a bidirectional Transformer encoder with modified embedding layers tailored for capturing item information in recommendation datasets. We apply masking to text sentences and use them to predict the representations of the unmasked sentences, helping the model learn generalizable item embeddings. To further improve recommendation performance and language understanding, we employ a two-stage training strategy incorporating self-supervised learning losses. Experiments on six real-world datasets demonstrate that JEPA4Rec consistently outperforms state-of-the-art methods, particularly in cross-domain, cross-platform, and low-resource scenarios. 

**Abstract (ZH)**: JEPA4Rec：结合项文本描述的语言建模的联合嵌入预测架构 

---
# Leveraging Auto-Distillation and Generative Self-Supervised Learning in Residual Graph Transformers for Enhanced Recommender Systems 

**Title (ZH)**: 利用自动蒸馏和生成自监督学习在残差图变换器中的应用以增强推荐系统 

**Authors**: Eya Mhedhbi, Youssef Mourchid, Alice Othmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.10500)  

**Abstract**: This paper introduces a cutting-edge method for enhancing recommender systems through the integration of generative self-supervised learning (SSL) with a Residual Graph Transformer. Our approach emphasizes the importance of superior data enhancement through the use of pertinent pretext tasks, automated through rationale-aware SSL to distill clear ways of how users and items interact. The Residual Graph Transformer incorporates a topology-aware transformer for global context and employs residual connections to improve graph representation learning. Additionally, an auto-distillation process refines self-supervised signals to uncover consistent collaborative rationales. Experimental evaluations on multiple datasets demonstrate that our approach consistently outperforms baseline methods. 

**Abstract (ZH)**: 基于残差图变换器的生成自监督学习增强推荐系统方法 

---
# MTCNET: Multi-task Learning Paradigm for Crowd Count Estimation 

**Title (ZH)**: MTCNET：多任务学习框架用于人群计数估计 

**Authors**: Abhay Kumar, Nishant Jain, Suraj Tripathi, Chirag Singh, Kamal Krishna  

**Link**: [PDF](https://arxiv.org/pdf/1908.08652)  

**Abstract**: We propose a Multi-Task Learning (MTL) paradigm based deep neural network architecture, called MTCNet (Multi-Task Crowd Network) for crowd density and count estimation. Crowd count estimation is challenging due to the non-uniform scale variations and the arbitrary perspective of an individual image. The proposed model has two related tasks, with Crowd Density Estimation as the main task and Crowd-Count Group Classification as the auxiliary task. The auxiliary task helps in capturing the relevant scale-related information to improve the performance of the main task. The main task model comprises two blocks: VGG-16 front-end for feature extraction and a dilated Convolutional Neural Network for density map generation. The auxiliary task model shares the same front-end as the main task, followed by a CNN classifier. Our proposed network achieves 5.8% and 14.9% lower Mean Absolute Error (MAE) than the state-of-the-art methods on ShanghaiTech dataset without using any data augmentation. Our model also outperforms with 10.5% lower MAE on UCF_CC_50 dataset. 

**Abstract (ZH)**: 基于多任务学习的众人群体密度和计数估算的MTCNet架构 

---
# RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users 

**Title (ZH)**: RealWebAssist：基于真实用户的长视角网页助手基准 

**Authors**: Suyu Ye, Haojun Shi, Darren Shih, Hyokun Yun, Tanya Roosta, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10445)  

**Abstract**: To achieve successful assistance with long-horizon web-based tasks, AI agents must be able to sequentially follow real-world user instructions over a long period. Unlike existing web-based agent benchmarks, sequential instruction following in the real world poses significant challenges beyond performing a single, clearly defined task. For instance, real-world human instructions can be ambiguous, require different levels of AI assistance, and may evolve over time, reflecting changes in the user's mental state. To address this gap, we introduce RealWebAssist, a novel benchmark designed to evaluate sequential instruction-following in realistic scenarios involving long-horizon interactions with the web, visual GUI grounding, and understanding ambiguous real-world user instructions. RealWebAssist includes a dataset of sequential instructions collected from real-world human users. Each user instructs a web-based assistant to perform a series of tasks on multiple websites. A successful agent must reason about the true intent behind each instruction, keep track of the mental state of the user, understand user-specific routines, and ground the intended tasks to actions on the correct GUI elements. Our experimental results show that state-of-the-art models struggle to understand and ground user instructions, posing critical challenges in following real-world user instructions for long-horizon web assistance. 

**Abstract (ZH)**: 实现长周期网络任务的有效辅助，AI代理必须能够在长期内按照现实世界用户的指令顺序执行。与现有的基于网络的代理基准不同，真实世界的顺序指令遵循带来了超出执行单一明确任务之外的重大挑战。例如，现实世界的用户指令可能是模糊的，需要不同程度的AI辅助，并且可能会随时间演变，反映用户心理状态的变化。为了弥补这一空白，我们引入了RealWebAssist，这是一个新颖的基准，旨在评估在涉及长时间与网络交互、视觉GUI定位以及理解模糊的现实世界用户指令的现实场景中顺序指令遵循的能力。RealWebAssist 包括从真实世界人类用户收集的顺序指令数据集。每个用户指示一个基于网络的助手在多个网站上执行一系列任务。成功的代理必须理解每条指令背后的真正意图，跟踪用户的心理状态，了解用户的特定程序，并将意图任务与正确的GUI元素上的操作进行关联。我们的实验结果表明，最先进的模型难以理解并定位用户指令，在长时间网络辅助中遵循现实世界的用户指令面临关键挑战。 

---
