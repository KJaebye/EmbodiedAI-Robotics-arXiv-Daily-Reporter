# R2RGEN: Real-to-Real 3D Data Generation for Spatially Generalized Manipulation 

**Title (ZH)**: R2RGEN: 实景到实景的3D数据生成及其在空间泛化操作中的应用 

**Authors**: Xiuwei Xu, Angyuan Ma, Hankun Li, Bingyao Yu, Zheng Zhu, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08547)  

**Abstract**: Towards the aim of generalized robotic manipulation, spatial generalization is the most fundamental capability that requires the policy to work robustly under different spatial distribution of objects, environment and agent itself. To achieve this, substantial human demonstrations need to be collected to cover different spatial configurations for training a generalized visuomotor policy via imitation learning. Prior works explore a promising direction that leverages data generation to acquire abundant spatially diverse data from minimal source demonstrations. However, most approaches face significant sim-to-real gap and are often limited to constrained settings, such as fixed-base scenarios and predefined camera viewpoints. In this paper, we propose a real-to-real 3D data generation framework (R2RGen) that directly augments the pointcloud observation-action pairs to generate real-world data. R2RGen is simulator- and rendering-free, thus being efficient and plug-and-play. Specifically, given a single source demonstration, we introduce an annotation mechanism for fine-grained parsing of scene and trajectory. A group-wise augmentation strategy is proposed to handle complex multi-object compositions and diverse task constraints. We further present camera-aware processing to align the distribution of generated data with real-world 3D sensor. Empirically, R2RGen substantially enhances data efficiency on extensive experiments and demonstrates strong potential for scaling and application on mobile manipulation. 

**Abstract (ZH)**: 向通用机器人 manipulation 的目标迈进，空间泛化是最基本的能力，要求策略在不同物体分布、环境和自身位置的情况下都能稳健工作。为了实现这一目标，需要收集大量的人类演示数据以涵盖不同的空间配置，通过模仿学习训练通用的视觉-运动策略。此前的工作探索了一条有希望的方向，利用数据生成从少量源演示中获取丰富的空间多样数据。然而，大多数方法面临着显著的仿真实验到真实环境的差距，并且通常局限于约束场景，如固定基座场景和预定义的摄像头视角。在本文中，我们提出了一种实时到实时的3D数据生成框架（R2RGen），它可以直接增强点云观测-动作对以生成真实世界数据。R2RGen 是无需模拟器和渲染的，因此高效且即插即用。具体来说，给定单一源演示，我们引入了一种标注机制进行细粒度的场景和轨迹解析。我们提出了一种组别增强策略来处理复杂的多物体组合和多样的任务约束。进一步地，我们提出了摄像头感知的处理方法，以使生成数据的分布与真实世界的3D传感器分布相匹配。实验结果表明，R2RGen 显著提高了数据效率，并展示了在移动操作中扩展和应用的强潜力。 

---
# VeMo: A Lightweight Data-Driven Approach to Model Vehicle Dynamics 

**Title (ZH)**: VeMo: 一种轻量级的数据驱动车辆动力学建模方法 

**Authors**: Girolamo Oddo, Roberto Nuca, Matteo Parsani  

**Link**: [PDF](https://arxiv.org/pdf/2510.07447)  

**Abstract**: Developing a dynamic model for a high-performance vehicle is a complex problem that requires extensive structural information about the system under analysis. This information is often unavailable to those who did not design the vehicle and represents a typical issue in autonomous driving applications, which are frequently developed on top of existing vehicles; therefore, vehicle models are developed under conditions of information scarcity. This paper proposes a lightweight encoder-decoder model based on Gate Recurrent Unit layers to correlate the vehicle's future state with its past states, measured onboard, and control actions the driver performs. The results demonstrate that the model achieves a maximum mean relative error below 2.6% in extreme dynamic conditions. It also shows good robustness when subject to noisy input data across the interested frequency components. Furthermore, being entirely data-driven and free from physical constraints, the model exhibits physical consistency in the output signals, such as longitudinal and lateral accelerations, yaw rate, and the vehicle's longitudinal velocity. 

**Abstract (ZH)**: 基于门循环单元的轻量级编码解码器模型用于高performance车辆的状态预测 

---
# Have We Scene It All? Scene Graph-Aware Deep Point Cloud Compression 

**Title (ZH)**: 我们见过所有的场景吗？基于场景图的深度点云压缩 

**Authors**: Nikolaos Stathoulopoulos, Christoforos Kanellakis, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.08512)  

**Abstract**: Efficient transmission of 3D point cloud data is critical for advanced perception in centralized and decentralized multi-agent robotic systems, especially nowadays with the growing reliance on edge and cloud-based processing. However, the large and complex nature of point clouds creates challenges under bandwidth constraints and intermittent connectivity, often degrading system performance. We propose a deep compression framework based on semantic scene graphs. The method decomposes point clouds into semantically coherent patches and encodes them into compact latent representations with semantic-aware encoders conditioned by Feature-wise Linear Modulation (FiLM). A folding-based decoder, guided by latent features and graph node attributes, enables structurally accurate reconstruction. Experiments on the SemanticKITTI and nuScenes datasets show that the framework achieves state-of-the-art compression rates, reducing data size by up to 98% while preserving both structural and semantic fidelity. In addition, it supports downstream applications such as multi-robot pose graph optimization and map merging, achieving trajectory accuracy and map alignment comparable to those obtained with raw LiDAR scans. 

**Abstract (ZH)**: 基于语义场景图的高效3D点云数据传输对于集中式和分布式多机器人系统高级感知至关重要，尤其是在现今广泛依赖边缘和基于云的处理的情况下。然而，点云的大型和复杂性质在带宽受限和间歇连接环境下造成了挑战，通常会降低系统性能。我们提出了一种基于语义场景图的深度压缩框架。该方法将点云分解为语义上一致的patches，并通过特征aware编码器（基于特征向量线性调制FiLM）将它们编码为紧凑的潜在表示。基于折叠的解码器在潜在特征和图节点属性的指导下，实现结构上准确的重构。在SemanticKITTI和nuScenes数据集上的实验表明，该框架实现了最先进的压缩率，即使数据量减少了高达98%，也能保持结构和语义的保真度。此外，该框架支持多机器人位姿图优化和地图合并等下游应用，其轨迹精度和地图对齐与原始LiDAR扫描相当。 

---
# Kontinuous Kontext: Continuous Strength Control for Instruction-based Image Editing 

**Title (ZH)**: 连续.context：基于指令的图像编辑中的连续强度控制 

**Authors**: Rishubh Parihar, Or Patashnik, Daniil Ostashev, R. Venkatesh Babu, Daniel Cohen-Or, Kuan-Chieh Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08532)  

**Abstract**: Instruction-based image editing offers a powerful and intuitive way to manipulate images through natural language. Yet, relying solely on text instructions limits fine-grained control over the extent of edits. We introduce Kontinuous Kontext, an instruction-driven editing model that provides a new dimension of control over edit strength, enabling users to adjust edits gradually from no change to a fully realized result in a smooth and continuous manner. Kontinuous Kontext extends a state-of-the-art image editing model to accept an additional input, a scalar edit strength which is then paired with the edit instruction, enabling explicit control over the extent of the edit. To inject this scalar information, we train a lightweight projector network that maps the input scalar and the edit instruction to coefficients in the model's modulation space. For training our model, we synthesize a diverse dataset of image-edit-instruction-strength quadruplets using existing generative models, followed by a filtering stage to ensure quality and consistency. Kontinuous Kontext provides a unified approach for fine-grained control over edit strength for instruction driven editing from subtle to strong across diverse operations such as stylization, attribute, material, background, and shape changes, without requiring attribute-specific training. 

**Abstract (ZH)**: 基于指令的图像编辑提供了一种通过自然语言操控图像的强大而直观的方式。然而，仅依赖文本指令限制了对编辑程度的精细控制。我们引入了Kontinuous Kontext，一种以指令驱动的编辑模型，提供了控制编辑强度的新维度，使用户能够从无更改到完全实现结果以平滑连续的方式逐步调整编辑。Kontinuous Kontext 将最先进的图像编辑模型扩展为接受额外输入——一个标量编辑强度，然后将该标量编辑强度与编辑指令配对，从而可以对编辑的程度进行显式控制。为了注入这种标量信息，我们训练了一个轻量级的投影网络，该网络将输入标量和编辑指令映射到模型调制空间中的系数。在训练我们的模型时，我们使用现有的生成模型合成了图像编辑指令强度四元组的多样化数据集，并通过过滤阶段确保质量和一致性。Kontinuous Kontext 提供了一种统一的方法，可以在细微到强烈的范围内对指令驱动编辑的各种操作（如风格化、属性、材质、背景和形状变化）的编辑强度进行精细控制，而无需特定属性的训练。 

---
# AI-Driven Radiology Report Generation for Traumatic Brain Injuries 

**Title (ZH)**: AI驱动的放射学报告生成在创伤性脑损伤中的应用 

**Authors**: Riadh Bouslimi, Houda Trabelsi, Wahiba Ben Abdssalem Karaa, Hana Hedhli  

**Link**: [PDF](https://arxiv.org/pdf/2510.08498)  

**Abstract**: Traumatic brain injuries present significant diagnostic challenges in emergency medicine, where the timely interpretation of medical images is crucial for patient outcomes. In this paper, we propose a novel AI-based approach for automatic radiology report generation tailored to cranial trauma cases. Our model integrates an AC-BiFPN with a Transformer architecture to capture and process complex medical imaging data such as CT and MRI scans. The AC-BiFPN extracts multi-scale features, enabling the detection of intricate anomalies like intracranial hemorrhages, while the Transformer generates coherent, contextually relevant diagnostic reports by modeling long-range dependencies. We evaluate the performance of our model on the RSNA Intracranial Hemorrhage Detection dataset, where it outperforms traditional CNN-based models in both diagnostic accuracy and report generation. This solution not only supports radiologists in high-pressure environments but also provides a powerful educational tool for trainee physicians, offering real-time feedback and enhancing their learning experience. Our findings demonstrate the potential of combining advanced feature extraction with transformer-based text generation to improve clinical decision-making in the diagnosis of traumatic brain injuries. 

**Abstract (ZH)**: 创伤性脑损伤在急诊医学中给诊断带来了重大挑战，及时解读医学影像对于患者预后至关重要。本文提出了一种针对颅脑创伤病例的新型基于AI的自动放射学报告生成方法。该模型结合了AC-BiFPN与Transformer架构，以捕获和处理如CT和MRI扫描等复杂的医学影像数据。AC-BiFPN提取多尺度特征，能够检测复杂的异常，如颅内出血，而Transformer则通过建模长距离依赖关系生成连贯且与上下文相关的诊断报告。我们在RSNA颅内出血检测数据集上评估了该模型的性能，结果显示其在诊断准确性和报告生成方面均优于传统的基于CNN的模型。该解决方案不仅在高压环境中支持放射科医生，还为实习生提供了一个强大的教学工具，提供实时反馈，增强他们的学习体验。我们的研究结果表明，将高级特征提取与基于Transformer的文本生成结合，有望改善创伤性脑损伤诊断中的临床决策。 

---
# Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception 

**Title (ZH)**: 评价小规模 vision-language 模型在距离依赖性交通感知中的表现 

**Authors**: Nikos Theodoridis, Tim Brophy, Reenu Mohandas, Ganesh Sistu, Fiachra Collins, Anthony Scanlan, Ciaran Eising  

**Link**: [PDF](https://arxiv.org/pdf/2510.08352)  

**Abstract**: Vision-Language Models (VLMs) are becoming increasingly powerful, demonstrating strong performance on a variety of tasks that require both visual and textual understanding. Their strong generalisation abilities make them a promising component for automated driving systems, which must handle unexpected corner cases. However, to be trusted in such safety-critical applications, a model must first possess a reliable perception system. Moreover, since critical objects and agents in traffic scenes are often at a distance, we require systems that are not "shortsighted", i.e., systems with strong perception capabilities at both close (up to 20 meters) and long (30+ meters) range. With this in mind, we introduce Distance-Annotated Traffic Perception Question Answering (DTPQA), the first Visual Question Answering (VQA) benchmark focused solely on perception-based questions in traffic scenes, enriched with distance annotations. By excluding questions that require reasoning, we ensure that model performance reflects perception capabilities alone. Since automated driving hardware has limited processing power and cannot support large VLMs, our study centers on smaller VLMs. More specifically, we evaluate several state-of-the-art (SOTA) small VLMs on DTPQA and show that, despite the simplicity of the questions, these models significantly underperform compared to humans (~60% average accuracy for the best-performing small VLM versus ~85% human performance). However, it is important to note that the human sample size was relatively small, which imposes statistical limitations. We also identify specific perception tasks, such as distinguishing left from right, that remain particularly challenging for these models. 

**Abstract (ZH)**: Distance-Annotated Traffic Perception Question Answering 

---
# Learning Neural Exposure Fields for View Synthesis 

**Title (ZH)**: 学习神经曝光场进行视图合成 

**Authors**: Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Christina Tsalicoglou, Keisuke Tateno, Jonathan T. Barron, Federico Tombari  

**Link**: [PDF](https://arxiv.org/pdf/2510.08279)  

**Abstract**: Recent advances in neural scene representations have led to unprecedented quality in 3D reconstruction and view synthesis. Despite achieving high-quality results for common benchmarks with curated data, outputs often degrade for data that contain per image variations such as strong exposure changes, present, e.g., in most scenes with indoor and outdoor areas or rooms with windows. In this paper, we introduce Neural Exposure Fields (NExF), a novel technique for robustly reconstructing 3D scenes with high quality and 3D-consistent appearance from challenging real-world captures. In the core, we propose to learn a neural field predicting an optimal exposure value per 3D point, enabling us to optimize exposure along with the neural scene representation. While capture devices such as cameras select optimal exposure per image/pixel, we generalize this concept and perform optimization in 3D instead. This enables accurate view synthesis in high dynamic range scenarios, bypassing the need of post-processing steps or multi-exposure captures. Our contributions include a novel neural representation for exposure prediction, a system for joint optimization of the scene representation and the exposure field via a novel neural conditioning mechanism, and demonstrated superior performance on challenging real-world data. We find that our approach trains faster than prior works and produces state-of-the-art results on several benchmarks improving by over 55% over best-performing baselines. 

**Abstract (ZH)**: Recent Advances in Neural Scene Representations: Neural Exposure Fields for Robust 3D Reconstruction and View Synthesis in Challenging Real-World Scenarios 

---
# TTOM: Test-Time Optimization and Memorization for Compositional Video Generation 

**Title (ZH)**: TTOM：测试时优化和记忆化在组件视频生成中的应用 

**Authors**: Leigang Qu, Ziyang Wang, Na Zheng, Wenjie Wang, Liqiang Nie, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2510.07940)  

**Abstract**: Video Foundation Models (VFMs) exhibit remarkable visual generation performance, but struggle in compositional scenarios (e.g., motion, numeracy, and spatial relation). In this work, we introduce Test-Time Optimization and Memorization (TTOM), a training-free framework that aligns VFM outputs with spatiotemporal layouts during inference for better text-image alignment. Rather than direct intervention to latents or attention per-sample in existing work, we integrate and optimize new parameters guided by a general layout-attention objective. Furthermore, we formulate video generation within a streaming setting, and maintain historical optimization contexts with a parametric memory mechanism that supports flexible operations, such as insert, read, update, and delete. Notably, we found that TTOM disentangles compositional world knowledge, showing powerful transferability and generalization. Experimental results on the T2V-CompBench and Vbench benchmarks establish TTOM as an effective, practical, scalable, and efficient framework to achieve cross-modal alignment for compositional video generation on the fly. 

**Abstract (ZH)**: Test-Time Optimization and Memorization for Better Text-Video Alignment in Compositional Video Generation 

---
# UltraLED: Learning to See Everything in Ultra-High Dynamic Range Scenes 

**Title (ZH)**: UltraLED：在超高动态范围场景中学习全面观测 

**Authors**: Yuang Meng, Xin Jin, Lina Lei, Chun-Le Guo, Chongyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.07741)  

**Abstract**: Ultra-high dynamic range (UHDR) scenes exhibit significant exposure disparities between bright and dark regions. Such conditions are commonly encountered in nighttime scenes with light sources. Even with standard exposure settings, a bimodal intensity distribution with boundary peaks often emerges, making it difficult to preserve both highlight and shadow details simultaneously. RGB-based bracketing methods can capture details at both ends using short-long exposure pairs, but are susceptible to misalignment and ghosting artifacts. We found that a short-exposure image already retains sufficient highlight detail. The main challenge of UHDR reconstruction lies in denoising and recovering information in dark regions. In comparison to the RGB images, RAW images, thanks to their higher bit depth and more predictable noise characteristics, offer greater potential for addressing this challenge. This raises a key question: can we learn to see everything in UHDR scenes using only a single short-exposure RAW image? In this study, we rely solely on a single short-exposure frame, which inherently avoids ghosting and motion blur, making it particularly robust in dynamic scenes. To achieve that, we introduce UltraLED, a two-stage framework that performs exposure correction via a ratio map to balance dynamic range, followed by a brightness-aware RAW denoiser to enhance detail recovery in dark regions. To support this setting, we design a 9-stop bracketing pipeline to synthesize realistic UHDR images and contribute a corresponding dataset based on diverse scenes, using only the shortest exposure as input for reconstruction. Extensive experiments show that UltraLED significantly outperforms existing single-frame approaches. Our code and dataset are made publicly available at this https URL. 

**Abstract (ZH)**: 超高的动态范围(UHDR)场景在明亮区域和暗淡区域之间表现出显著的曝光差异。这种条件在包含光源的夜间场景中常见。即使使用标准的曝光设置，亮度分布通常会出现双峰边界，使得同时保存高光和阴影细节变得困难。基于RGB的曝光级联方法可以使用短曝光-长曝光对来捕获两端的细节，但易受到对齐错误和鬼影伪影的影响。我们发现，短曝光图像本身已保留足够的高光细节。UHDR重建的主要挑战在于暗区噪声去除和信息恢复。与RGB图像相比，RAW图像由于具有更高的位深度和更可预测的噪声特性，为解决这一挑战提供了更大的潜力。这提出了一个核心问题：我们能否仅使用单张短曝光RAW图像就能学会看到UHDR场景中的所有细节？在本研究中，我们仅依赖于一张短曝光帧，从根本上避免了鬼影和运动模糊，使其在动态场景中尤为 robust。为了实现这一点，我们引入了UltraLED，这是一种两阶段框架，通过比率图执行曝光校正以平衡动态范围，随后是一个亮度感知的RAW降噪器，以增强暗区的细节恢复。为支持这一设置，我们设计了一个9级级联管道来合成真实的UHDR图像，并基于多样场景贡献了一个相应的数据集，仅使用最短曝光作为重建输入。广泛的实验表明，UltraLED显著优于现有的单帧方法。我们的代码和数据集已公开在以下链接：这个 https URL。 

---
# Curriculum Learning with Synthetic Data for Enhanced Pulmonary Nodule Detection in Chest Radiographs 

**Title (ZH)**: 基于合成数据的课程学习方法以增强胸部X光片中肺结节检测 

**Authors**: Pranav Sambhu, Om Guin, Madhav Sambhu, Jinho Cha  

**Link**: [PDF](https://arxiv.org/pdf/2510.07681)  

**Abstract**: This study evaluates whether integrating curriculum learning with diffusion-based synthetic augmentation can enhance the detection of difficult pulmonary nodules in chest radiographs, particularly those with low size, brightness, and contrast, which often challenge conventional AI models due to data imbalance and limited annotation. A Faster R-CNN with a Feature Pyramid Network (FPN) backbone was trained on a hybrid dataset comprising expert-labeled NODE21 (1,213 patients; 52.4 percent male; mean age 63.2 +/- 11.5 years), VinDr-CXR, CheXpert, and 11,206 DDPM-generated synthetic images. Difficulty scores based on size, brightness, and contrast guided curriculum learning. Performance was compared to a non-curriculum baseline using mean average precision (mAP), Dice score, and area under the curve (AUC). Statistical tests included bootstrapped confidence intervals, DeLong tests, and paired t-tests. The curriculum model achieved a mean AUC of 0.95 versus 0.89 for the baseline (p < 0.001), with improvements in sensitivity (70 percent vs. 48 percent) and accuracy (82 percent vs. 70 percent). Stratified analysis demonstrated consistent gains across all difficulty bins (Easy to Very Hard). Grad-CAM visualizations confirmed more anatomically focused attention under curriculum learning. These results suggest that curriculum-guided synthetic augmentation enhances model robustness and generalization for pulmonary nodule detection. 

**Abstract (ZH)**: 本研究评估了将 Curriculum 学习与扩散基础的合成增强结合是否能提高胸部X光片中难以检测的肺结节检测能力，特别是那些尺寸小、亮度低、对比度低的结节，由于数据不平衡和有限的注释，这些结节经常挑战传统的AI模型。研究使用包含专家标注的NODE21（1,213例患者；男性占52.4%；平均年龄63.2±11.5岁）、VinDr-CXR、CheXpert及11,206张DDPM生成的合成图像的混合数据集对带有特征金字塔网络（FPN）骨干的Faster R-CNN进行了训练。根据尺寸、亮度和对比度的难度评分指导Curriculum学习。性能与无Curriculum基线进行了比较，使用平均平均精度（mAP）、Dice分数和曲线下面积（AUC）进行评估。统计测试包括置信区间 bootstrap、DeLong检验和配对t检验。Curriculum模型的平均AUC为0.95，而基线为0.89（p < 0.001），并在灵敏度和准确性方面表现出改进（分别从48%提高到70%，从70%提高到82%）。分层分析表明，Curriculum学习在所有难度级别上均显示出一致的收益。Grad-CAM可视化确认了Curriculum学习下更专注于解剖结构的注意力。这些结果表明，由Curriculum指导的合成增强可以提高肺结节检测模型的鲁棒性和泛化能力。 

---
# Controllable Video Synthesis via Variational Inference 

**Title (ZH)**: 基于变分推断的可控视频合成 

**Authors**: Haoyi Duan, Yunzhi Zhang, Yilun Du, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07670)  

**Abstract**: Many video workflows benefit from a mixture of user controls with varying granularity, from exact 4D object trajectories and camera paths to coarse text prompts, while existing video generative models are typically trained for fixed input formats. We develop a video synthesis method that addresses this need and generates samples with high controllability for specified elements while maintaining diversity for under-specified ones. We cast the task as variational inference to approximate a composed distribution, leveraging multiple video generation backbones to account for all task constraints collectively. To address the optimization challenge, we break down the problem into step-wise KL divergence minimization over an annealed sequence of distributions, and further propose a context-conditioned factorization technique that reduces modes in the solution space to circumvent local optima. Experiments suggest that our method produces samples with improved controllability, diversity, and 3D consistency compared to prior works. 

**Abstract (ZH)**: 许多视频工作流需要不同程度的用户控制，从精确的4D对象轨迹和摄像机路径到粗略的文字提示，而现有的视频生成模型通常仅针对固定的输入格式进行训练。我们开发了一种视频合成方法，以满足这一需求，该方法在指定元素上生成高可控性样本，同时在欠指定的元素上保持多样性。我们将任务视为变分推断，以逼近组合分布，并利用多个视频生成骨干来联合考虑所有任务约束。为了解决优化挑战，我们将问题分解为逐步最小化退火序列分布的KL散度，并提出了一种基于上下文的因子分解技术，以减少解空间中的模式，从而规避局部最优。实验结果表明，与现有方法相比，我们的方法能够生成具有更好可控性、多样性和3D一致性的样本。 

---
# TCIP: Threshold-Controlled Iterative Pyramid Network for Deformable Medical Image Registration 

**Title (ZH)**: TCIP：阈值控制迭代金字塔网络在可变形医学图像配准中的应用 

**Authors**: Heming Wu, Di Wang, Tai Ma, Peng Zhao, Yubin Xiao, Zhongke Wu, Xing-Ce Wang, Chuang Li, Xuan Wu, You Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.07666)  

**Abstract**: Although pyramid networks have demonstrated superior performance in deformable medical image registration, their decoder architectures are inherently prone to propagating and accumulating anatomical structure misalignments. Moreover, most existing models do not adaptively determine the number of iterations for optimization under varying deformation requirements across images, resulting in either premature termination or excessive iterations that degrades registration accuracy. To effectively mitigate the accumulation of anatomical misalignments, we propose the Feature-Enhanced Residual Module (FERM) as the core component of each decoding layer in the pyramid network. FERM comprises three sequential blocks that extract anatomical semantic features, learn to suppress irrelevant features, and estimate the final deformation field, respectively. To adaptively determine the number of iterations for varying images, we propose the dual-stage Threshold-Controlled Iterative (TCI) strategy. In the first stage, TCI assesses registration stability and with asserted stability, it continues with the second stage to evaluate convergence. We coin the model that integrates FERM and TCI as Threshold-Controlled Iterative Pyramid (TCIP). Extensive experiments on three public brain MRI datasets and one abdomen CT dataset demonstrate that TCIP outperforms the state-of-the-art (SOTA) registration networks in terms of accuracy, while maintaining comparable inference speed and a compact model parameter size. Finally, we assess the generalizability of FERM and TCI by integrating them with existing registration networks and further conduct ablation studies to validate the effectiveness of these two proposed methods. 

**Abstract (ZH)**: 虽然 Pyramid 网络在可变形医学图像配准中表现出色，但其解码架构天生容易传播和累积解剖结构错位。此外，大多数现有模型不能根据不同图像的变形要求自适应地确定优化的迭代次数，导致要么过早终止要么进行过多迭代，从而降低配准精度。为了有效减轻解剖结构错位的累积，我们提出了一种特征增强残差模块（FERM）作为 Pyramid 网络中每一层解码器的核心组件。FERM 包含三个连续的块，分别提取解剖语义特征、学习抑制无关特征以及估计最终变形场。为了根据不同图像自适应地确定迭代次数，我们提出了双阶段阈值控制迭代（TCI）策略。在第一阶段，TCI 评估配准的稳定性，在确认稳定性后，进入第二阶段评估收敛性。我们将集成了 FERM 和 TCI 的模型命名为阈值控制迭代 Pyramid（TCIP）。在三个公开的脑 MRI 数据集和一个腹部 CT 数据集上进行的广泛实验表明，TCIP 在准确性上优于当前最先进的（SOTA）配准网络，同时保持类似的推理速度和紧凑的模型参数量。最后，我们通过将 FERM 和 TCI 与现有配准网络集成，并进一步进行消融研究，评估了这两种方法的一般适用性。 

---
# Mitigating Surgical Data Imbalance with Dual-Prediction Video Diffusion Model 

**Title (ZH)**: 使用双预测视频扩散模型缓解手术数据不平衡问题 

**Authors**: Danush Kumar Venkatesh, Adam Schmidt, Muhammad Abdullah Jamal, Omid Mohareri  

**Link**: [PDF](https://arxiv.org/pdf/2510.07345)  

**Abstract**: Surgical video datasets are essential for scene understanding, enabling procedural modeling and intra-operative support. However, these datasets are often heavily imbalanced, with rare actions and tools under-represented, which limits the robustness of downstream models. We address this challenge with $SurgiFlowVid$, a sparse and controllable video diffusion framework for generating surgical videos of under-represented classes. Our approach introduces a dual-prediction diffusion module that jointly denoises RGB frames and optical flow, providing temporal inductive biases to improve motion modeling from limited samples. In addition, a sparse visual encoder conditions the generation process on lightweight signals (e.g., sparse segmentation masks or RGB frames), enabling controllability without dense annotations. We validate our approach on three surgical datasets across tasks including action recognition, tool presence detection, and laparoscope motion prediction. Synthetic data generated by our method yields consistent gains of 10-20% over competitive baselines, establishing $SurgiFlowVid$ as a promising strategy to mitigate data imbalance and advance surgical video understanding methods. 

**Abstract (ZH)**: 手术视频数据集对于场景理解、程序建模和术中支持至关重要。然而，这些数据集往往严重失衡，稀有动作和工具的代表性不足，这限制了下游模型的鲁棒性。我们通过提出一种稀疏可控的视频扩散框架$SurgiFlowVid$来应对这一挑战，该框架能够生成代表性不足类别的手术视频。我们的方法引入了双预测扩散模块，该模块联合去噪RGB帧和光学流，提供时间上的归纳偏差，以提高从有限样本中建模运动的能力。此外，稀疏视觉编码器通过轻量级信号（如稀疏分割掩码或RGB帧）来条件生成过程，从而在无需密集标注的情况下实现可控性。我们在三个手术数据集上对包括动作识别、工具存在检测和腹腔镜运动预测等任务进行了验证。由我们的方法生成的合成数据在多个基准上的表现提高了10-20%，这表明$SurgiFlowVid$是一种有前景的策略，可以缓解数据不平衡问题并推进手术视频理解方法。 

---
