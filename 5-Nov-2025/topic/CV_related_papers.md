# From the Laboratory to Real-World Application: Evaluating Zero-Shot Scene Interpretation on Edge Devices for Mobile Robotics 

**Title (ZH)**: 从实验室到实际应用：在移动机器人边缘设备上评估零样本场景解释 

**Authors**: Nicolas Schuler, Lea Dewald, Nick Baldig, Jürgen Graf  

**Link**: [PDF](https://arxiv.org/pdf/2511.02427)  

**Abstract**: Video Understanding, Scene Interpretation and Commonsense Reasoning are highly challenging tasks enabling the interpretation of visual information, allowing agents to perceive, interact with and make rational decisions in its environment. Large Language Models (LLMs) and Visual Language Models (VLMs) have shown remarkable advancements in these areas in recent years, enabling domain-specific applications as well as zero-shot open vocabulary tasks, combining multiple domains. However, the required computational complexity poses challenges for their application on edge devices and in the context of Mobile Robotics, especially considering the trade-off between accuracy and inference time. In this paper, we investigate the capabilities of state-of-the-art VLMs for the task of Scene Interpretation and Action Recognition, with special regard to small VLMs capable of being deployed to edge devices in the context of Mobile Robotics. The proposed pipeline is evaluated on a diverse dataset consisting of various real-world cityscape, on-campus and indoor scenarios. The experimental evaluation discusses the potential of these small models on edge devices, with particular emphasis on challenges, weaknesses, inherent model biases and the application of the gained information. Supplementary material is provided via the following repository: this https URL 

**Abstract (ZH)**: 视频理解、场景解释和常识推理是高度具有挑战性的任务，使视觉信息的解释成为可能，使智能体能够在其环境中感知、交互并做出理性的决策。大型语言模型（LLMs）和视觉语言模型（VLMs）近年来在这些领域取得了显著的进步，不仅实现了领域特定的应用，还能完成零样本的开放词汇任务，结合了多个领域。然而，所需的计算复杂性对这些模型在边缘设备上的应用以及在移动机器人领域的应用构成了挑战，特别是在准确性和推理时间之间的权衡方面。在本文中，我们研究了最先进的VLMs在场景解释和动作识别任务中的能力，特别关注能够在移动机器人领域部署到边缘设备的较小VLMs。提出的管道在包含各种真实世界城市景观、校园和室内场景的多样化数据集上进行了评估。实验评估讨论了这些小型模型在边缘设备上的潜力，特别是针对挑战、弱点、模型固有的偏差以及获得的信息的应用进行了重点讨论。补充材料可通过以下repository获取：this https URL。 

---
# Synthetic Crop-Weed Image Generation and its Impact on Model Generalization 

**Title (ZH)**: 合成作物-杂草图像生成及其对模型泛化能力的影响 

**Authors**: Garen Boyadjian, Cyrille Pierre, Johann Laconte, Riccardo Bertoglio  

**Link**: [PDF](https://arxiv.org/pdf/2511.02417)  

**Abstract**: Precise semantic segmentation of crops and weeds is necessary for agricultural weeding robots. However, training deep learning models requires large annotated datasets, which are costly to obtain in real fields. Synthetic data can reduce this burden, but the gap between simulated and real images remains a challenge. In this paper, we present a pipeline for procedural generation of synthetic crop-weed images using Blender, producing annotated datasets under diverse conditions of plant growth, weed density, lighting, and camera angle. We benchmark several state-of-the-art segmentation models on synthetic and real datasets and analyze their cross-domain generalization. Our results show that training on synthetic images leads to a sim-to-real gap of 10%, surpassing previous state-of-the-art methods. Moreover, synthetic data demonstrates good generalization properties, outperforming real datasets in cross-domain scenarios. These findings highlight the potential of synthetic agricultural datasets and support hybrid strategies for more efficient model training. 

**Abstract (ZH)**: 作物和杂草的精准语义分割对于农业除草机器人至关重要。然而，训练深度学习模型需要大量的标注数据集，而在实际田地中获得这些数据的成本很高。合成数据可以减轻这一负担，但模拟和真实图像之间的差距仍然是一个挑战。在本文中，我们提出了一种使用Blender进行程序化生成合成作物-杂草图像的管道，生成在不同植物生长条件、杂草密度、光照和相机角度下的标注数据集。我们在合成数据集和真实数据集上benchmark了几种最先进的分割模型，并分析了它们在跨领域泛化的性能。我们的结果表明，使用合成图像进行训练导致的模拟到现实世界的差距为10%，超过了之前的最先进的方法。此外，合成数据展示了良好的跨领域泛化特性，在跨领域场景中优于真实数据集。这些发现突显了合成农业数据集的潜力，并支持混合策略以实现更高效的模型训练。 

---
# Wireless Video Semantic Communication with Decoupled Diffusion Multi-frame Compensation 

**Title (ZH)**: 无线视频语义通信与解耦扩散多帧补偿 

**Authors**: Bingyan Xie, Yongpeng Wu, Yuxuan Shi, Biqian Feng, Wenjun Zhang, Jihong Park, Tony Quek  

**Link**: [PDF](https://arxiv.org/pdf/2511.02478)  

**Abstract**: Existing wireless video transmission schemes directly conduct video coding in pixel level, while neglecting the inner semantics contained in videos. In this paper, we propose a wireless video semantic communication framework with decoupled diffusion multi-frame compensation (DDMFC), abbreviated as WVSC-D, which integrates the idea of semantic communication into wireless video transmission scenarios. WVSC-D first encodes original video frames as semantic frames and then conducts video coding based on such compact representations, enabling the video coding in semantic level rather than pixel level. Moreover, to further reduce the communication overhead, a reference semantic frame is introduced to substitute motion vectors of each frame in common video coding methods. At the receiver, DDMFC is proposed to generate compensated current semantic frame by a two-stage conditional diffusion process. With both the reference frame transmission and DDMFC frame compensation, the bandwidth efficiency improves with satisfying video transmission performance. Experimental results verify the performance gain of WVSC-D over other DL-based methods e.g. DVSC about 1.8 dB in terms of PSNR. 

**Abstract (ZH)**: 无线视频语义通信框架WVSC-D：解耦扩散多帧补偿 

---
# Purrturbed but Stable: Human-Cat Invariant Representations Across CNNs, ViTs and Self-Supervised ViTs 

**Title (ZH)**: 扰动下的稳定：跨CNN、ViT和自监督ViT的人猫不变表示 

**Authors**: Arya Shah, Vaibhav Tripathi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02404)  

**Abstract**: Cats and humans differ in ocular anatomy. Most notably, Felis Catus (domestic cats) have vertically elongated pupils linked to ambush predation; yet, how such specializations manifest in downstream visual representations remains incompletely understood. We present a unified, frozen-encoder benchmark that quantifies feline-human cross-species representational alignment in the wild, across convolutional networks, supervised Vision Transformers, windowed transformers, and self-supervised ViTs (DINO), using layer-wise Centered Kernel Alignment (linear and RBF) and Representational Similarity Analysis, with additional distributional and stability tests reported in the paper. Across models, DINO ViT-B/16 attains the most substantial alignment (mean CKA-RBF $\approx0.814$, mean CKA-linear $\approx0.745$, mean RSA $\approx0.698$), peaking at early blocks, indicating that token-level self-supervision induces early-stage features that bridge species-specific statistics. Supervised ViTs are competitive on CKA yet show weaker geometric correspondence than DINO (e.g., ViT-B/16 RSA $\approx0.53$ at block8; ViT-L/16 $\approx0.47$ at block14), revealing depth-dependent divergences between similarity and representational geometry. CNNs remain strong baselines but below plain ViTs on alignment, and windowed transformers underperform plain ViTs, implicating architectural inductive biases in cross-species alignment. Results indicate that self-supervision coupled with ViT inductive biases yields representational geometries that more closely align feline and human visual systems than widely used CNNs and windowed Transformers, providing testable neuroscientific hypotheses about where and how cross-species visual computations converge. We release our code and dataset for reference and reproducibility. 

**Abstract (ZH)**: 猫和人类在眼部解剖结构上存在差异。最显著的是，Felis catus（家猫）具有与伏击捕食相关的垂直拉长瞳孔；然而，此类特化如何在下游视觉表征中表现仍然知之甚少。我们提供了一个统一的冻结编码器基准，使用逐层中心核对齐（线性和RBF）和表征相似性分析，在野生条件下量化猫与人类跨物种的表征对齐，覆盖卷积网络、监督视觉变换器、窗口变换器以及自监督ViTs（DINO），并在论文中报告了分布性和稳定性测试。在各种模型中，DINO ViT-B/16取得最显著的对齐（均值CKA-RBF ≈0.814，均值CKA-线性 ≈0.745，均值RSA ≈0.698），在早期块中达到峰值，表明 tokenize 级别自监督诱导出能跨越物种特异性统计数据的早期阶段特征。监督变换器在CKA方面具有竞争力，但在几何对应方面弱于DINO（例如，ViT-B/16 在块8的RSA ≈0.53；ViT-L/16 在块14的RSA ≈0.47），揭示了相似性和表征几何之间的深度依赖性差异。卷积神经网络作为基准模型仍然很强，但与简单的变换器相比，在对齐上较低，窗口变换器的表现也低于简单的变换器，暗示了交叉物种对齐在架构诱导偏见中的作用。结果表明，结合自监督与变换器诱导偏见可以产生更接近猫和人类视觉系统的表征几何结构，优于广泛使用的卷积神经网络和窗口变换器，提供测试性的神经科学假设，关于跨物种视觉计算在何处以及如何收敛。我们发布了我们的代码和数据集供参考和复现。 

---
# Object-Centric 3D Gaussian Splatting for Strawberry Plant Reconstruction and Phenotyping 

**Title (ZH)**: 基于对象的草莓植物3D高斯绘制重建与表型分析 

**Authors**: Jiajia Li, Keyi Zhu, Qianwen Zhang, Dong Chen, Qi Sun, Zhaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.02207)  

**Abstract**: Strawberries are among the most economically significant fruits in the United States, generating over $2 billion in annual farm-gate sales and accounting for approximately 13% of the total fruit production value. Plant phenotyping plays a vital role in selecting superior cultivars by characterizing plant traits such as morphology, canopy structure, and growth dynamics. However, traditional plant phenotyping methods are time-consuming, labor-intensive, and often destructive. Recently, neural rendering techniques, notably Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have emerged as powerful frameworks for high-fidelity 3D reconstruction. By capturing a sequence of multi-view images or videos around a target plant, these methods enable non-destructive reconstruction of complex plant architectures. Despite their promise, most current applications of 3DGS in agricultural domains reconstruct the entire scene, including background elements, which introduces noise, increases computational costs, and complicates downstream trait analysis. To address this limitation, we propose a novel object-centric 3D reconstruction framework incorporating a preprocessing pipeline that leverages the Segment Anything Model v2 (SAM-2) and alpha channel background masking to achieve clean strawberry plant reconstructions. This approach produces more accurate geometric representations while substantially reducing computational time. With a background-free reconstruction, our algorithm can automatically estimate important plant traits, such as plant height and canopy width, using DBSCAN clustering and Principal Component Analysis (PCA). Experimental results show that our method outperforms conventional pipelines in both accuracy and efficiency, offering a scalable and non-destructive solution for strawberry plant phenotyping. 

**Abstract (ZH)**: 美国草莓是最具经济意义的水果之一，年农场门市销售额超过20亿美元，占总水果生产价值的约13%。植物表型在选择优良品种方面起着关键作用，通过表征形态、冠层结构和生长动态等植物特征。然而，传统的植物表型方法耗时、劳动密集且往往具有破坏性。最近，神经渲染技术，尤其是神经辐射场（NeRF）和3D高斯点积（3DGS），已成为高保真3D重建的强大框架。通过拍摄目标植物周围的多视角图像或视频序列，这些方法能够实现非破坏性的复杂植物结构重建。尽管前景可人，但当前大多数3DGS在农业领域的应用都会重建整个场景，包括背景元素，这引入了噪声、增加了计算成本，并增加了后续特征分析的复杂性。为解决这一限制，我们提出了一种新的以对象为中心的3D重建框架，结合了利用Segment Anything Model v2（SAM-2）和alpha通道背景遮罩的预处理管道，以实现清晰的草莓植物重建。该方法提供了更准确的几何表示，并显著减少了计算时间。通过无背景的重建，我们的算法可以自动使用DBSCAN聚类和主成分分析（PCA）估计重要植物特征，如植物高度和冠层宽度。实验结果显示，我们的方法在准确性与效率上均优于传统管道，提供了一种可扩展且非破坏性的草莓植物表型解决方案。 

---
# MM-UNet: Morph Mamba U-shaped Convolutional Networks for Retinal Vessel Segmentation 

**Title (ZH)**: MM-UNet：形态morph曼巴U形卷积网络用于视网膜血管分割 

**Authors**: Jiawen Liu, Yuanbo Zeng, Jiaming Liang, Yizhen Yang, Yiheng Zhang, Enhui Cai, Xiaoqi Sheng, Hongmin Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.02193)  

**Abstract**: Accurate detection of retinal vessels plays a critical role in reflecting a wide range of health status indicators in the clinical diagnosis of ocular diseases. Recently, advances in deep learning have led to a surge in retinal vessel segmentation methods, which have significantly contributed to the quantitative analysis of vascular morphology. However, retinal vasculature differs significantly from conventional segmentation targets in that it consists of extremely thin and branching structures, whose global morphology varies greatly across images. These characteristics continue to pose challenges to segmentation precision and robustness. To address these issues, we propose MM-UNet, a novel architecture tailored for efficient retinal vessel segmentation. The model incorporates Morph Mamba Convolution layers, which replace pointwise convolutions to enhance branching topological perception through morph, state-aware feature sampling. Additionally, Reverse Selective State Guidance modules integrate reverse guidance theory with state-space modeling to improve geometric boundary awareness and decoding efficiency. Extensive experiments conducted on two public retinal vessel segmentation datasets demonstrate the superior performance of the proposed method in segmentation accuracy. Compared to the existing approaches, MM-UNet achieves F1-score gains of 1.64 $\%$ on DRIVE and 1.25 $\%$ on STARE, demonstrating its effectiveness and advancement. The project code is public via this https URL. 

**Abstract (ZH)**: 准确检测视网膜血管在眼科疾病的临床诊断中反映了广泛的健康状态指标，对于反映多种健康状态指标至关重要。近期，深度学习的进步推动了视网膜血管分割方法的显著发展，极大地促进了血管形态的定量分析。然而，视网膜血管与传统分割目标显著不同，由极其细薄且分支的结构组成，其全局形态在不同图像中差异极大。这些特性继续对分割精度和鲁棒性构成挑战。为了解决这些问题，我们提出MM-UNet，一种专门用于高效视网膜血管分割的新架构。该模型采用Morph Mamba卷积层，用形态感知特征采样取代点wise卷积，增强分支拓扑感知。此外，Reverse Selective State Guidance模块结合逆向引导理论和状态空间建模，提高了几何边界意识和解码效率。在两个公开的视网膜血管分割数据集上的广泛实验表明，所提出的方法在分割准确性上表现出优异性能。与现有方法相比，MM-UNet在DRIVE上的F1得分提高了1.64%，在STARE上提高了1.25%，展示了其有效性和先进性。项目代码可通过此链接公开获取。 

---
# Dynamic Population Distribution Aware Human Trajectory Generation with Diffusion Model 

**Title (ZH)**: 动态人口分布aware的人类轨迹生成方法——基于扩散模型 

**Authors**: Qingyue Long, Can Rong, Tong Li, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.01929)  

**Abstract**: Human trajectory data is crucial in urban planning, traffic engineering, and public health. However, directly using real-world trajectory data often faces challenges such as privacy concerns, data acquisition costs, and data quality. A practical solution to these challenges is trajectory generation, a method developed to simulate human mobility behaviors. Existing trajectory generation methods mainly focus on capturing individual movement patterns but often overlook the influence of population distribution on trajectory generation. In reality, dynamic population distribution reflects changes in population density across different regions, significantly impacting individual mobility behavior. Thus, we propose a novel trajectory generation framework based on a diffusion model, which integrates the dynamic population distribution constraints to guide high-fidelity generation outcomes. Specifically, we construct a spatial graph to enhance the spatial correlation of trajectories. Then, we design a dynamic population distribution aware denoising network to capture the spatiotemporal dependencies of human mobility behavior as well as the impact of population distribution in the denoising process. Extensive experiments show that the trajectories generated by our model can resemble real-world trajectories in terms of some critical statistical metrics, outperforming state-of-the-art algorithms by over 54%. 

**Abstract (ZH)**: 人类轨迹数据在城市规划、交通工程和公共卫生中的应用至关重要。然而，直接使用真实世界轨迹数据常常面临着隐私顾虑、数据采集成本和数据质量等方面的挑战。一种实用的解决方案是轨迹生成，这种方法用于模拟人类移动行为。现有的轨迹生成方法主要侧重于捕捉个体移动模式，但往往忽略了人口分布对轨迹生成的影响。实际上，动态的人口分布反映了不同地区人口密度的变化，对个体移动行为产生了显著影响。因此，我们提出了一种基于扩散模型的新型轨迹生成框架，该框架整合了动态人口分布约束，以指导高保真生成结果。具体来说，我们构建了一个空间图来增强轨迹的空间相关性。然后，我们设计了一个动态人口分布感知的去噪网络，以捕捉人类移动行为的时空依赖性和去噪过程中的人口分布影响。大量的实验表明，由我们模型生成的轨迹在某些关键统计指标上与真实世界轨迹相似，并且在性能上超过了最先进的算法超过54%。 

---
