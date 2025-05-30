# Explainable Scene Understanding with Qualitative Representations and Graph Neural Networks 

**Title (ZH)**: 基于定性表示和图神经网络的可解释场景理解 

**Authors**: Nassim Belmecheri, Arnaud Gotlieb, Nadjib Lazaar, Helge Spieker  

**Link**: [PDF](https://arxiv.org/pdf/2504.12817)  

**Abstract**: This paper investigates the integration of graph neural networks (GNNs) with Qualitative Explainable Graphs (QXGs) for scene understanding in automated driving. Scene understanding is the basis for any further reactive or proactive decision-making. Scene understanding and related reasoning is inherently an explanation task: why is another traffic participant doing something, what or who caused their actions? While previous work demonstrated QXGs' effectiveness using shallow machine learning models, these approaches were limited to analysing single relation chains between object pairs, disregarding the broader scene context. We propose a novel GNN architecture that processes entire graph structures to identify relevant objects in traffic scenes. We evaluate our method on the nuScenes dataset enriched with DriveLM's human-annotated relevance labels. Experimental results show that our GNN-based approach achieves superior performance compared to baseline methods. The model effectively handles the inherent class imbalance in relevant object identification tasks while considering the complete spatial-temporal relationships between all objects in the scene. Our work demonstrates the potential of combining qualitative representations with deep learning approaches for explainable scene understanding in autonomous driving systems. 

**Abstract (ZH)**: 本文研究了图神经网络（GNNs）与定性可解释图（QXGs）在自动驾驶中场景理解中的集成应用。场景理解是任何进一步反应性或前瞻性决策的基础。场景理解和相关推理本质上是一种解释任务：为什么另一个交通参与者会采取某种行动，是什么或谁引起了他们的行为？尽管先前的工作展示了QXGs在浅层机器学习模型中的有效性，但这些方法仅限于分析对象对之间的单一关系链，忽略了更广泛的情景上下文。我们提出了一种新的GNN架构，用于处理整个图结构以识别交通场景中的相关对象。我们在带有DriveLM的人标注相关性标签的nuScenes数据集上评估了该方法。实验结果表明，基于GNN的方法在基准方法上取得了更好的性能。该模型有效处理了相关对象识别任务中的固有类别不平衡问题，同时考虑了场景中所有对象的完整空间-时间关系。我们的工作展示了将定性表示与深度学习方法结合以实现可解释的自动驾驶系统场景理解的潜力。 

---
# 3D-PNAS: 3D Industrial Surface Anomaly Synthesis with Perlin Noise 

**Title (ZH)**: 3D-PNAS：使用Perlin噪声的3D工业表面 anomaly 合成 

**Authors**: Yifeng Cheng, Juan Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.12856)  

**Abstract**: Large pretrained vision foundation models have shown significant potential in various vision tasks. However, for industrial anomaly detection, the scarcity of real defect samples poses a critical challenge in leveraging these models. While 2D anomaly generation has significantly advanced with established generative models, the adoption of 3D sensors in industrial manufacturing has made leveraging 3D data for surface quality inspection an emerging trend. In contrast to 2D techniques, 3D anomaly generation remains largely unexplored, limiting the potential of 3D data in industrial quality inspection. To address this gap, we propose a novel yet simple 3D anomaly generation method, 3D-PNAS, based on Perlin noise and surface parameterization. Our method generates realistic 3D surface anomalies by projecting the point cloud onto a 2D plane, sampling multi-scale noise values from a Perlin noise field, and perturbing the point cloud along its normal direction. Through comprehensive visualization experiments, we demonstrate how key parameters - including noise scale, perturbation strength, and octaves, provide fine-grained control over the generated anomalies, enabling the creation of diverse defect patterns from pronounced deformations to subtle surface variations. Additionally, our cross-category experiments show that the method produces consistent yet geometrically plausible anomalies across different object types, adapting to their specific surface characteristics. We also provide a comprehensive codebase and visualization toolkit to facilitate future research. 

**Abstract (ZH)**: 基于珀林噪声和表面参数化的3D异常生成方法3D-PNAS 

---
# UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control 

**Title (ZH)**: UniPhys：基于扩散的统一规划与控制器，实现灵活的物理_basis角色控制 

**Authors**: Yan Wu, Korrawe Karunratanakul, Zhengyi Luo, Siyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12540)  

**Abstract**: Generating natural and physically plausible character motion remains challenging, particularly for long-horizon control with diverse guidance signals. While prior work combines high-level diffusion-based motion planners with low-level physics controllers, these systems suffer from domain gaps that degrade motion quality and require task-specific fine-tuning. To tackle this problem, we introduce UniPhys, a diffusion-based behavior cloning framework that unifies motion planning and control into a single model. UniPhys enables flexible, expressive character motion conditioned on multi-modal inputs such as text, trajectories, and goals. To address accumulated prediction errors over long sequences, UniPhys is trained with the Diffusion Forcing paradigm, learning to denoise noisy motion histories and handle discrepancies introduced by the physics simulator. This design allows UniPhys to robustly generate physically plausible, long-horizon motions. Through guided sampling, UniPhys generalizes to a wide range of control signals, including unseen ones, without requiring task-specific fine-tuning. Experiments show that UniPhys outperforms prior methods in motion naturalness, generalization, and robustness across diverse control tasks. 

**Abstract (ZH)**: 基于扩散的行为克隆框架UniPhys：统一运动规划与控制生成自然且物理合理的角色运动仍然具有挑战性，尤其是在具有多样指导信号的长期控制中。尽管现有工作结合了高层的基于扩散的运动规划器与低层的物理控制器，但这些系统存在领域差距，这会降低运动质量并需要特定任务的微调。为解决这一问题，我们引入了UniPhys，这是一种基于扩散的行为克隆框架，将运动规划与控制统一到一个模型中。UniPhys能够根据多模态输入（如文本、轨迹和目标）生成灵活且具表达性的角色运动。为了解决长时间序列中累积的预测误差，UniPhys采用扩散强迫范式进行训练，学习去除噪声的运动历史并处理物理模拟器引入的不一致性。这种设计使UniPhys能够稳健地生成物理合理的长期运动。通过引导采样，UniPhys能够在无需特定任务微调的情况下泛化到广泛的控制信号，包括未见过的信号。实验结果表明，UniPhys在运动自然性、泛化能力和跨多种控制任务的鲁棒性方面优于之前的方法。 

---
# PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding 

**Title (ZH)**: PerceptionLM：开放访问的数据和模型 for 详细视觉理解 

**Authors**: Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, Miguel Martin, Huiyu Wang, Hanoona Rasheed, Peize Sun, Po-Yao Huang, Daniel Bolya, Nikhila Ravi, Shashank Jain, Tammy Stark, Shane Moon, Babak Damavandi, Vivian Lee, Andrew Westbury, Salman Khan, Philipp Krähenbühl, Piotr Dollár, Lorenzo Torresani, Kristen Grauman, Christoph Feichtenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2504.13180)  

**Abstract**: Vision-language models are integral to computer vision research, yet many high-performing models remain closed-source, obscuring their data, design and training recipe. The research community has responded by using distillation from black-box models to label training data, achieving strong benchmark results, at the cost of measurable scientific progress. However, without knowing the details of the teacher model and its data sources, scientific progress remains difficult to measure. In this paper, we study building a Perception Language Model (PLM) in a fully open and reproducible framework for transparent research in image and video understanding. We analyze standard training pipelines without distillation from proprietary models and explore large-scale synthetic data to identify critical data gaps, particularly in detailed video understanding. To bridge these gaps, we release 2.8M human-labeled instances of fine-grained video question-answer pairs and spatio-temporally grounded video captions. Additionally, we introduce PLM-VideoBench, a suite for evaluating challenging video understanding tasks focusing on the ability to reason about "what", "where", "when", and "how" of a video. We make our work fully reproducible by providing data, training recipes, code & models. 

**Abstract (ZH)**: 视觉语言模型是计算机视觉研究中的重要组成部分，但许多高性能模型仍为封闭源代码，遮蔽了其数据、设计和训练方法。研究社区通过从黑盒模型中提取知识标记训练数据，实现了强有力的基准结果，但牺牲了可量化的科学进步。然而，缺乏了解教师模型及其数据源的细节，科学进步仍难以衡量。在本文中，我们研究了在开放和可重复的框架下构建感知语言模型（PLM），以推动图像和视频理解的透明研究。我们分析了不依赖专有模型的知识提取的标准训练流水线，并探索大规模合成数据以识别关键的数据缺口，特别是针对详细的视频理解。为填补这些缺口，我们发布了280万个人标注的细粒度视频问答实例和时空定位的视频描述。此外，我们引入了PLM-VideoBench，这是一个评估视频理解任务的套件，重点关注对视频中“什么”、“哪里”、“何时”和“如何”的推理能力。我们通过提供数据、训练方法、代码和模型来使我们的工作完全可重复。 

---
# $\texttt{Complex-Edit}$: CoT-Like Instruction Generation for Complexity-Controllable Image Editing Benchmark 

**Title (ZH)**: $\texttt{Complex-Edit}$: 基于CoT-like指令生成的复杂度可控图片编辑基准 

**Authors**: Siwei Yang, Mude Hui, Bingchen Zhao, Yuyin Zhou, Nataniel Ruiz, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.13143)  

**Abstract**: We introduce $\texttt{Complex-Edit}$, a comprehensive benchmark designed to systematically evaluate instruction-based image editing models across instructions of varying complexity. To develop this benchmark, we harness GPT-4o to automatically collect a diverse set of editing instructions at scale. Our approach follows a well-structured ``Chain-of-Edit'' pipeline: we first generate individual atomic editing tasks independently and then integrate them to form cohesive, complex instructions. Additionally, we introduce a suite of metrics to assess various aspects of editing performance, along with a VLM-based auto-evaluation pipeline that supports large-scale assessments. Our benchmark yields several notable insights: 1) Open-source models significantly underperform relative to proprietary, closed-source models, with the performance gap widening as instruction complexity increases; 2) Increased instructional complexity primarily impairs the models' ability to retain key elements from the input images and to preserve the overall aesthetic quality; 3) Decomposing a complex instruction into a sequence of atomic steps, executed in a step-by-step manner, substantially degrades performance across multiple metrics; 4) A straightforward Best-of-N selection strategy improves results for both direct editing and the step-by-step sequential approach; and 5) We observe a ``curse of synthetic data'': when synthetic data is involved in model training, the edited images from such models tend to appear increasingly synthetic as the complexity of the editing instructions rises -- a phenomenon that intriguingly also manifests in the latest GPT-4o outputs. 

**Abstract (ZH)**: 我们介绍了一个综合基准$\texttt{Complex-Edit}$，用于系统地评估基于指令的图像编辑模型在不同复杂度指令下的性能。为了开发这一基准，我们利用GPT-4o自动收集了大量多样化的编辑指令。我们的方法遵循一个精心设计的“编辑链”管道：首先独立生成个体的原子编辑任务，然后将其整合形成连贯且复杂的指令。此外，我们还引入了一套评估编辑性能各个方面的度量标准，并提供了一种基于VLM的自动评估管道，支持大规模评估。基准测试提供了几个重要的见解：1）开源模型相对于闭源的专有模型显著性能较低，随着指令复杂性的增加，性能差距逐渐扩大；2）指令复杂性的增加主要影响模型保持输入图像的关键要素和保留整体视觉质量的能力；3）将复杂的指令分解为一系列原子步骤，逐步执行，会显著在多个度量标准上降低性能；4）简单的Best-of-N选择策略能改善直接编辑和逐步顺序方法的结果；5）我们观察到“合成数据的诅咒”：当合成数据参与模型训练时，随着编辑指令复杂性的增加，从这些模型生成的编辑图像倾向于越来越具合成性——这一现象在最新的GPT-4o输出中也有所体现。 

---
# NTIRE 2025 Challenge on Short-form UGC Video Quality Assessment and Enhancement: Methods and Results 

**Title (ZH)**: NTIRE 2025挑战赛：短形式UGC视频质量评估与增强的方法与结果 

**Authors**: Xin Li, Kun Yuan, Bingchen Li, Fengbin Guan, Yizhen Shao, Zihao Yu, Xijun Wang, Yiting Lu, Wei Luo, Suhang Yao, Ming Sun, Chao Zhou, Zhibo Chen, Radu Timofte, Yabin Zhang, Ao-Xiang Zhang, Tianwu Zhi, Jianzhao Liu, Yang Li, Jingwen Xu, Yiting Liao, Yushen Zuo, Mingyang Wu, Renjie Li, Shengyun Zhong, Zhengzhong Tu, Yufan Liu, Xiangguang Chen, Zuowei Cao, Minhao Tang, Shan Liu, Kexin Zhang, Jingfen Xie, Yan Wang, Kai Chen, Shijie Zhao, Yunchen Zhang, Xiangkai Xu, Hong Gao, Ji Shi, Yiming Bao, Xiugang Dong, Xiangsheng Zhou, Yaofeng Tu, Ying Liang, Yiwen Wang, Xinning Chai, Yuxuan Zhang, Zhengxue Cheng, Yingsheng Qin, Yucai Yang, Rong Xie, Li Song, Wei Sun, Kang Fu, Linhan Cao, Dandan Zhu, Kaiwei Zhang, Yucheng Zhu, Zicheng Zhang, Menghan Hu, Xiongkuo Min, Guangtao Zhai, Zhi Jin, Jiawei Wu, Wei Wang, Wenjian Zhang, Yuhai Lan, Gaoxiong Yi, Hengyuan Na, Wang Luo, Di Wu, MingYin Bai, Jiawang Du, Zilong Lu, Zhenyu Jiang, Hui Zeng, Ziguan Cui, Zongliang Gan, Guijin Tang, Xinglin Xie, Kehuan Song, Xiaoqiang Lu, Licheng Jiao, Fang Liu, Xu Liu, Puhua Chen, Ha Thu Nguyen, Katrien De Moor, Seyed Ali Amirshahi, Mohamed-Chaker Larabi, Qi Tang, Linfeng He, Zhiyong Gao, Zixuan Gao, Guohua Zhang, Zhiye Huang, Yi Deng, Qingmiao Jiang, Lu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13131)  

**Abstract**: This paper presents a review for the NTIRE 2025 Challenge on Short-form UGC Video Quality Assessment and Enhancement. The challenge comprises two tracks: (i) Efficient Video Quality Assessment (KVQ), and (ii) Diffusion-based Image Super-Resolution (KwaiSR). Track 1 aims to advance the development of lightweight and efficient video quality assessment (VQA) models, with an emphasis on eliminating reliance on model ensembles, redundant weights, and other computationally expensive components in the previous IQA/VQA competitions. Track 2 introduces a new short-form UGC dataset tailored for single image super-resolution, i.e., the KwaiSR dataset. It consists of 1,800 synthetically generated S-UGC image pairs and 1,900 real-world S-UGC images, which are split into training, validation, and test sets using a ratio of 8:1:1. The primary objective of the challenge is to drive research that benefits the user experience of short-form UGC platforms such as Kwai and TikTok. This challenge attracted 266 participants and received 18 valid final submissions with corresponding fact sheets, significantly contributing to the progress of short-form UGC VQA and image superresolution. The project is publicly available at this https URL ChallengeCVPR-NTIRE2025. 

**Abstract (ZH)**: NTIRE 2025 挑战赛：短形式UGC视频质量评估与增强综述 

---
# Science-T2I: Addressing Scientific Illusions in Image Synthesis 

**Title (ZH)**: 科学-图像合成：解决图像合成中的科学错觉 

**Authors**: Jialuo Li, Wenhao Chai, Xingyu Fu, Haiyang Xu, Saining Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.13129)  

**Abstract**: We present a novel approach to integrating scientific knowledge into generative models, enhancing their realism and consistency in image synthesis. First, we introduce Science-T2I, an expert-annotated adversarial dataset comprising adversarial 20k image pairs with 9k prompts, covering wide distinct scientific knowledge categories. Leveraging Science-T2I, we present SciScore, an end-to-end reward model that refines the assessment of generated images based on scientific knowledge, which is achieved by augmenting both the scientific comprehension and visual capabilities of pre-trained CLIP model. Additionally, based on SciScore, we propose a two-stage training framework, comprising a supervised fine-tuning phase and a masked online fine-tuning phase, to incorporate scientific knowledge into existing generative models. Through comprehensive experiments, we demonstrate the effectiveness of our framework in establishing new standards for evaluating the scientific realism of generated content. Specifically, SciScore attains performance comparable to human-level, demonstrating a 5% improvement similar to evaluations conducted by experienced human evaluators. Furthermore, by applying our proposed fine-tuning method to FLUX, we achieve a performance enhancement exceeding 50% on SciScore. 

**Abstract (ZH)**: 我们将一种将科学知识整合到生成模型中的新颖方法应用于图像合成，增强了生成图像的真实性和一致性。首先，我们介绍了Science-T2I，一个由专家注释的对抗性数据集，包含20,000个图像对和9,000个提示，涵盖广泛的科学知识类别。借助Science-T2I，我们提出了SciScore，这是一种端到端的奖励模型，基于科学知识精炼生成图像的评估，通过增强预训练CLIP模型的科学理解和视觉能力实现。此外，基于SciScore，我们提出了一个两阶段训练框架，包括监督微调阶段和掩码在线微调阶段，以将科学知识整合到现有的生成模型中。通过全面的实验，我们展示了该框架在评估生成内容的科学真实性方面建立新标准的有效性。具体而言，SciScore的表现达到与人类相当的水平，显示出与经验丰富的评估者进行评价类似的5%的改善。此外，通过对FLUX进行我们提出的微调方法的应用，在SciScore上的性能提升超过50%。 

---
# Enhancing Person-to-Person Virtual Try-On with Multi-Garment Virtual Try-Off 

**Title (ZH)**: 基于多 garments 虚拟试脱的 Personen-to-Person 虚拟试穿增强 

**Authors**: Riza Velioglu, Petra Bevandic, Robin Chan, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2504.13078)  

**Abstract**: Computer vision is transforming fashion through Virtual Try-On (VTON) and Virtual Try-Off (VTOFF). VTON generates images of a person in a specified garment using a target photo and a standardized garment image, while a more challenging variant, Person-to-Person Virtual Try-On (p2p-VTON), uses a photo of another person wearing the garment. VTOFF, on the other hand, extracts standardized garment images from clothed individuals. We introduce TryOffDiff, a diffusion-based VTOFF model. Built on a latent diffusion framework with SigLIP image conditioning, it effectively captures garment properties like texture, shape, and patterns. TryOffDiff achieves state-of-the-art results on VITON-HD and strong performance on DressCode dataset, covering upper-body, lower-body, and dresses. Enhanced with class-specific embeddings, it pioneers multi-garment VTOFF, the first of its kind. When paired with VTON models, it improves p2p-VTON by minimizing unwanted attribute transfer, such as skin color. Code is available at: this https URL 

**Abstract (ZH)**: 计算机视觉正在通过虚拟试穿（VTON）和虚拟脱下（VTOFF）改造时尚。VTON利用目标照片和标准服装图像生成一名人在指定服装中的图像，而更具挑战性的变体Person-to-Person Virtual Try-On（p2p-VTON）使用另一人在穿该服装的照片。VTOFF从穿着者身上提取标准服装图像。我们提出了TryOffDiff，这是一种基于扩散机制的VTOFF模型，通过潜空间扩散框架和SigLIP图像条件化，有效地捕捉服装的纹理、形状和图案特征。TryOffDiff在VITON-HD上取得了最先进的性能，并在涵盖上身、下身和连衣裙的DressCode数据集上表现出强大的性能。通过类别特定嵌入的增强，TryOffDiff开创了多件服装VTOFF，这是该领域的首个模型。当与VTON模型结合使用时，它可以减少不必要的属性转移，例如肤色。更多信息请参阅：this https URL。 

---
# Event-Enhanced Blurry Video Super-Resolution 

**Title (ZH)**: 事件增强模糊视频超分辨率 

**Authors**: Dachun Kai, Yueyi Zhang, Jin Wang, Zeyu Xiao, Zhiwei Xiong, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.13042)  

**Abstract**: In this paper, we tackle the task of blurry video super-resolution (BVSR), aiming to generate high-resolution (HR) videos from low-resolution (LR) and blurry inputs. Current BVSR methods often fail to restore sharp details at high resolutions, resulting in noticeable artifacts and jitter due to insufficient motion information for deconvolution and the lack of high-frequency details in LR frames. To address these challenges, we introduce event signals into BVSR and propose a novel event-enhanced network, Ev-DeblurVSR. To effectively fuse information from frames and events for feature deblurring, we introduce a reciprocal feature deblurring module that leverages motion information from intra-frame events to deblur frame features while reciprocally using global scene context from the frames to enhance event features. Furthermore, to enhance temporal consistency, we propose a hybrid deformable alignment module that fully exploits the complementary motion information from inter-frame events and optical flow to improve motion estimation in the deformable alignment process. Extensive evaluations demonstrate that Ev-DeblurVSR establishes a new state-of-the-art performance on both synthetic and real-world datasets. Notably, on real data, our method is +2.59 dB more accurate and 7.28$\times$ faster than the recent best BVSR baseline FMA-Net. Code: this https URL. 

**Abstract (ZH)**: 基于事件信号的模糊视频超分辨率：Ev-DeblurVSR 

---
# Prototypes are Balanced Units for Efficient and Effective Partially Relevant Video Retrieval 

**Title (ZH)**: 原型是高效的部分相关视频检索中的平衡单元 

**Authors**: WonJun Moon, Cheol-Ho Cho, Woojin Jun, Minho Shim, Taeoh Kim, Inwoong Lee, Dongyoon Wee, Jae-Pil Heo  

**Link**: [PDF](https://arxiv.org/pdf/2504.13035)  

**Abstract**: In a retrieval system, simultaneously achieving search accuracy and efficiency is inherently challenging. This challenge is particularly pronounced in partially relevant video retrieval (PRVR), where incorporating more diverse context representations at varying temporal scales for each video enhances accuracy but increases computational and memory costs. To address this dichotomy, we propose a prototypical PRVR framework that encodes diverse contexts within a video into a fixed number of prototypes. We then introduce several strategies to enhance text association and video understanding within the prototypes, along with an orthogonal objective to ensure that the prototypes capture a diverse range of content. To keep the prototypes searchable via text queries while accurately encoding video contexts, we implement cross- and uni-modal reconstruction tasks. The cross-modal reconstruction task aligns the prototypes with textual features within a shared space, while the uni-modal reconstruction task preserves all video contexts during encoding. Additionally, we employ a video mixing technique to provide weak guidance to further align prototypes and associated textual representations. Extensive evaluations on TVR, ActivityNet-Captions, and QVHighlights validate the effectiveness of our approach without sacrificing efficiency. 

**Abstract (ZH)**: 在检索系统中同时实现搜索准确性和效率本质上是具有挑战性的。这一挑战在部分相关视频检索（PRVR）中尤为突出，通过在每个视频中引入不同时间和尺度的多样性上下文表示可以提高准确性，但会增加计算和内存成本。为了解决这种二难境地，我们提出了一种原型PRVR框架，将视频中的多样性上下文编码为固定数量的原型。我们还引入了几种策略来增强原型内的文本关联和视频理解，并引入了一个正交目标以确保原型能够捕捉多样化的内容。为了在保持原型可通过文本查询检索的同时准确编码视频上下文，我们实现了跨模态和单模态重构任务。跨模态重构任务在共享空间中对齐原型和文本特征，而单模态重构任务在编码过程中保存所有视频上下文。此外，我们采用视频混音技术以提供弱指导，进一步对齐原型及其相关的文本表示。在TVR、ActivityNet-Captions和QVHighlights上的广泛评估验证了我们方法的有效性，而不牺牲效率。 

---
# Pose and Facial Expression Transfer by using StyleGAN 

**Title (ZH)**: 基于StyleGAN的表情和姿态转移 

**Authors**: Petr Jahoda, Jan Cech  

**Link**: [PDF](https://arxiv.org/pdf/2504.13021)  

**Abstract**: We propose a method to transfer pose and expression between face images. Given a source and target face portrait, the model produces an output image in which the pose and expression of the source face image are transferred onto the target identity. The architecture consists of two encoders and a mapping network that projects the two inputs into the latent space of StyleGAN2, which finally generates the output. The training is self-supervised from video sequences of many individuals. Manual labeling is not required. Our model enables the synthesis of random identities with controllable pose and expression. Close-to-real-time performance is achieved. 

**Abstract (ZH)**: 我们提出了一种在面部图像间transfer姿态和表情的方法。给定源面部肖像和目标面部身份，模型产生一个输出图像，在该图像中，源面部图像的姿态和表情被转移至目标面部身份上。该架构包含两个编码器和一个映射网络，将两个输入投影到StyleGAN2的潜在空间中，最终生成输出。训练是从多人的视频序列中进行自监督的，不需要手动标注。我们的模型能够合成具有可控姿态和表情的随机面部身份，并实现了接近实时的表现。 

---
# Image-Editing Specialists: An RLAIF Approach for Diffusion Models 

**Title (ZH)**: 图像编辑专家：基于RLAIF的方法在扩散模型中的应用 

**Authors**: Elior Benarous, Yilun Du, Heng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12833)  

**Abstract**: We present a novel approach to training specialized instruction-based image-editing diffusion models, addressing key challenges in structural preservation with input images and semantic alignment with user prompts. We introduce an online reinforcement learning framework that aligns the diffusion model with human preferences without relying on extensive human annotations or curating a large dataset. Our method significantly improves the realism and alignment with instructions in two ways. First, the proposed models achieve precise and structurally coherent modifications in complex scenes while maintaining high fidelity in instruction-irrelevant areas. Second, they capture fine nuances in the desired edit by leveraging a visual prompt, enabling detailed control over visual edits without lengthy textual prompts. This approach simplifies users' efforts to achieve highly specific edits, requiring only 5 reference images depicting a certain concept for training. Experimental results demonstrate that our models can perform intricate edits in complex scenes, after just 10 training steps. Finally, we showcase the versatility of our method by applying it to robotics, where enhancing the visual realism of simulated environments through targeted sim-to-real image edits improves their utility as proxies for real-world settings. 

**Abstract (ZH)**: 我们提出了一种新的基于指令的图像编辑扩散模型训练方法，解决了输入图像的结构保真和用户提示的语义对齐的关键挑战。我们引入了一种在线强化学习框架，无需依赖大量的人工注释或收集大数据集，即可使扩散模型与人类偏好对齐。我们的方法通过两种方式显著提高了真实感和指令对齐。首先，所提出的模型在复杂场景中实现了精细且结构一致的修改，同时在与指令无关的区域保持高度保真度。其次，通过利用视觉提示捕捉所需的细微修改，实现了对视觉编辑的详细控制，而无需冗长的文本提示。该方法简化了用户实现高度特定编辑的努力，只需5张表示某个概念的参考图像即可进行训练。实验结果表明，我们的模型仅经过10次训练步骤即可在复杂场景中执行复杂的编辑。最后，我们展示了我们方法的灵活性，将其应用于机器人领域，通过目标导向的模拟到现实的图像编辑来增强模拟环境的视觉真实感，从而提高它们作为现实世界替代品的实用性。 

---
# Hybrid Dense-UNet201 Optimization for Pap Smear Image Segmentation Using Spider Monkey Optimization 

**Title (ZH)**: 基于蜘蛛猴优化的Hybrid Dense-UNet20宫颈抹片图像分割优化 

**Authors**: Ach Khozaimi, Isnani Darti, Syaiful Anam, Wuryansari Muharini Kusumawinahyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12807)  

**Abstract**: Pap smear image segmentation is crucial for cervical cancer diagnosis. However, traditional segmentation models often struggle with complex cellular structures and variations in pap smear images. This study proposes a hybrid Dense-UNet201 optimization approach that integrates a pretrained DenseNet201 as the encoder for the U-Net architecture and optimizes it using the spider monkey optimization (SMO) algorithm. The Dense-UNet201 model excelled at feature extraction. The SMO was modified to handle categorical and discrete parameters. The SIPaKMeD dataset was used in this study and evaluated using key performance metrics, including loss, accuracy, Intersection over Union (IoU), and Dice coefficient. The experimental results showed that Dense-UNet201 outperformed U-Net, Res-UNet50, and Efficient-UNetB0. SMO Dense-UNet201 achieved a segmentation accuracy of 96.16%, an IoU of 91.63%, and a Dice coefficient score of 95.63%. These findings underscore the effectiveness of image preprocessing, pretrained models, and metaheuristic optimization in improving medical image analysis and provide new insights into cervical cell segmentation methods. 

**Abstract (ZH)**: Pap 疣片图像分割对于宫颈癌诊断至关重要。然而，传统分割模型往往难以处理 Pap 疣片图像中的复杂细胞结构和变化。本研究提出了一种结合预训练 DenseNet201 作为 U-Net 架构编码器并在其上使用蜘蛛猴优化（SMO）算法进行优化的混合 Dense-UNet201 优化方法。Dense-UNet201 模型在特征提取方面表现优异。SMO 被修改以处理类别和离散参数。本研究使用 SIPaKMeD 数据集，并使用包括损失、准确率、交并比 (IoU) 和 Dice 系数在内的关键性能指标进行评估。实验结果表明，Dense-UNet201 超过了 U-Net、Res-UNet50 和 Efficient-UNetB0。SMO 调整后的 Dense-UNet201 达到了 96.16% 的分割准确率、91.63% 的交并比和 95.63% 的 Dice 系数。这些发现强调了图像预处理、预训练模型和元启发式优化在提高医学图像分析方面的有效性，并为宫颈细胞分割方法提供了新的见解。 

---
# Set You Straight: Auto-Steering Denoising Trajectories to Sidestep Unwanted Concepts 

**Title (ZH)**: 正确定向：自动避开 unwanted 概念的去噪轨迹自引导 

**Authors**: Leyang Li, Shilin Lu, Yan Ren, Adams Wai-Kin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2504.12782)  

**Abstract**: Ensuring the ethical deployment of text-to-image models requires effective techniques to prevent the generation of harmful or inappropriate content. While concept erasure methods offer a promising solution, existing finetuning-based approaches suffer from notable limitations. Anchor-free methods risk disrupting sampling trajectories, leading to visual artifacts, while anchor-based methods rely on the heuristic selection of anchor concepts. To overcome these shortcomings, we introduce a finetuning framework, dubbed ANT, which Automatically guides deNoising Trajectories to avoid unwanted concepts. ANT is built on a key insight: reversing the condition direction of classifier-free guidance during mid-to-late denoising stages enables precise content modification without sacrificing early-stage structural integrity. This inspires a trajectory-aware objective that preserves the integrity of the early-stage score function field, which steers samples toward the natural image manifold, without relying on heuristic anchor concept selection. For single-concept erasure, we propose an augmentation-enhanced weight saliency map to precisely identify the critical parameters that most significantly contribute to the unwanted concept, enabling more thorough and efficient erasure. For multi-concept erasure, our objective function offers a versatile plug-and-play solution that significantly boosts performance. Extensive experiments demonstrate that ANT achieves state-of-the-art results in both single and multi-concept erasure, delivering high-quality, safe outputs without compromising the generative fidelity. Code is available at this https URL 

**Abstract (ZH)**: 确保文本到图像模型的伦理部署需要有效的技术来防止生成有害或不适当的内容。虽然概念擦除方法提供了有前景的解决方案，但现有的微调方法存在显著的限制。无锚方法存在破坏采样轨迹的风险，导致视觉 artifacts，而基于锚的方法依赖于启发式选择锚概念。为克服这些不足，我们引入了一种名为ANT的微调框架，它自动引导去噪轨迹以避免不必要的概念。ANT建立在一个关键洞察上：在中后期去噪阶段反向分类器无条件引导的条件方向能够实现精确的内容修改而不牺牲早期阶段的结构完整性。这启发了一个轨迹感知的目标，该目标保留了早期阶段得分函数域的完整性，引导样本向自然图像流形发展，而不依赖于启发式的锚概念选择。对于单一概念擦除，我们提出了一种增强增广的权重灵敏度图来精确识别对不需要的概念贡献最大的关键参数，从而实现更彻底和高效的擦除。对于多概念擦除，我们的目标函数提供了灵活的即插即用解决方案，显著提升了性能。大量实验表明，ANT在单概念和多概念擦除中均取得了最先进的结果，提供了高质量且安全的输出，而不牺牲生成保真度。代码可在以下链接获取。 

---
# TUMLS: Trustful Fully Unsupervised Multi-Level Segmentation for Whole Slide Images of Histology 

**Title (ZH)**: TUMLS: 可信的完全无监督多级分割方法用于病理学 Whole Slide Images 

**Authors**: Walid Rehamnia, Alexandra Getmanskaya, Evgeniy Vasilyev, Vadim Turlapov  

**Link**: [PDF](https://arxiv.org/pdf/2504.12718)  

**Abstract**: Digital pathology, augmented by artificial intelligence (AI), holds significant promise for improving the workflow of pathologists. However, challenges such as the labor-intensive annotation of whole slide images (WSIs), high computational demands, and trust concerns arising from the absence of uncertainty estimation in predictions hinder the practical application of current AI methodologies in histopathology. To address these issues, we present a novel trustful fully unsupervised multi-level segmentation methodology (TUMLS) for WSIs. TUMLS adopts an autoencoder (AE) as a feature extractor to identify the different tissue types within low-resolution training data. It selects representative patches from each identified group based on an uncertainty measure and then does unsupervised nuclei segmentation in their respective higher-resolution space without using any ML algorithms. Crucially, this solution integrates seamlessly into clinicians workflows, transforming the examination of a whole WSI into a review of concise, interpretable cross-level insights. This integration significantly enhances and accelerates the workflow while ensuring transparency. We evaluated our approach using the UPENN-GBM dataset, where the AE achieved a mean squared error (MSE) of 0.0016. Additionally, nucleus segmentation is assessed on the MoNuSeg dataset, outperforming all unsupervised approaches with an F1 score of 77.46% and a Jaccard score of 63.35%. These results demonstrate the efficacy of TUMLS in advancing the field of digital pathology. 

**Abstract (ZH)**: 基于人工智能增强的数字病理学：一种新型可信无监督多级分割方法（TUMLS）的研究 

---
# NTIRE 2025 Challenge on Day and Night Raindrop Removal for Dual-Focused Images: Methods and Results 

**Title (ZH)**: NTIRE 2025挑战赛：双焦距图像日间和夜间雨滴去除的方法与结果 

**Authors**: Xin Li, Yeying Jin, Xin Jin, Zongwei Wu, Bingchen Li, Yufei Wang, Wenhan Yang, Yu Li, Zhibo Chen, Bihan Wen, Robby T. Tan, Radu Timofte, Qiyu Rong, Hongyuan Jing, Mengmeng Zhang, Jinglong Li, Xiangyu Lu, Yi Ren, Yuting Liu, Meng Zhang, Xiang Chen, Qiyuan Guan, Jiangxin Dong, Jinshan Pan, Conglin Gou, Qirui Yang, Fangpu Zhang, Yunlong Lin, Sixiang Chen, Guoxi Huang, Ruirui Lin, Yan Zhang, Jingyu Yang, Huanjing Yue, Jiyuan Chen, Qiaosi Yi, Hongjun Wang, Chenxi Xie, Shuai Li, Yuhui Wu, Kaiyi Ma, Jiakui Hu, Juncheng Li, Liwen Pan, Guangwei Gao, Wenjie Li, Zhenyu Jin, Heng Guo, Zhanyu Ma, Yubo Wang, Jinghua Wang, Wangzhi Xing, Anjusree Karnavar, Diqi Chen, Mohammad Aminul Islam, Hao Yang, Ruikun Zhang, Liyuan Pan, Qianhao Luo, XinCao, Han Zhou, Yan Min, Wei Dong, Jun Chen, Taoyi Wu, Weijia Dou, Yu Wang, Shengjie Zhao, Yongcheng Huang, Xingyu Han, Anyan Huang, Hongtao Wu, Hong Wang, Yefeng Zheng, Abhijeet Kumar, Aman Kumar, Marcos V. Conde, Paula Garrido, Daniel Feijoo, Juan C. Benito, Guanglu Dong, Xin Lin, Siyuan Liu, Tianheng Zheng, Jiayu Zhong, Shouyi Wang, Xiangtai Li, Lanqing Guo, Lu Qi, Chao Ren, Shuaibo Wang, Shilong Zhang, Wanyu Zhou, Yunze Wu, Qinzhong Tan, Jieyuan Pei, Zhuoxuan Li, Jiayu Wang, Haoyu Bian, Haoran Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.12711)  

**Abstract**: This paper reviews the NTIRE 2025 Challenge on Day and Night Raindrop Removal for Dual-Focused Images. This challenge received a wide range of impressive solutions, which are developed and evaluated using our collected real-world Raindrop Clarity dataset. Unlike existing deraining datasets, our Raindrop Clarity dataset is more diverse and challenging in degradation types and contents, which includes day raindrop-focused, day background-focused, night raindrop-focused, and night background-focused degradations. This dataset is divided into three subsets for competition: 14,139 images for training, 240 images for validation, and 731 images for testing. The primary objective of this challenge is to establish a new and powerful benchmark for the task of removing raindrops under varying lighting and focus conditions. There are a total of 361 participants in the competition, and 32 teams submitting valid solutions and fact sheets for the final testing phase. These submissions achieved state-of-the-art (SOTA) performance on the Raindrop Clarity dataset. The project can be found at this https URL. 

**Abstract (ZH)**: NTIRE 2025日夜雨滴去除挑战评审：面向双焦图像的Raindrop Clarity数据集 

---
# Robo-SGG: Exploiting Layout-Oriented Normalization and Restitution for Robust Scene Graph Generation 

**Title (ZH)**: Robo-SGG：利用布局导向的规范化和重构实现稳健的场景图生成 

**Authors**: Changsheng Lv, Mengshi Qi, Zijian Fu, Huadong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.12606)  

**Abstract**: In this paper, we introduce a novel method named Robo-SGG, i.e., Layout-Oriented Normalization and Restitution for Robust Scene Graph Generation. Compared to the existing SGG setting, the robust scene graph generation aims to perform inference on a diverse range of corrupted images, with the core challenge being the domain shift between the clean and corrupted images. Existing SGG methods suffer from degraded performance due to compromised visual features e.g., corruption interference or occlusions. To obtain robust visual features, we exploit the layout information, which is domain-invariant, to enhance the efficacy of existing SGG methods on corrupted images. Specifically, we employ Instance Normalization(IN) to filter out the domain-specific feature and recover the unchangeable structural features, i.e., the positional and semantic relationships among objects by the proposed Layout-Oriented Restitution. Additionally, we propose a Layout-Embedded Encoder (LEE) that augments the existing object and predicate encoders within the SGG framework, enriching the robust positional and semantic features of objects and predicates. Note that our proposed Robo-SGG module is designed as a plug-and-play component, which can be easily integrated into any baseline SGG model. Extensive experiments demonstrate that by integrating the state-of-the-art method into our proposed Robo-SGG, we achieve relative improvements of 5.6%, 8.0%, and 6.5% in mR@50 for PredCls, SGCls, and SGDet tasks on the VG-C dataset, respectively, and achieve new state-of-the-art performance in corruption scene graph generation benchmark (VG-C and GQA-C). We will release our source code and model. 

**Abstract (ZH)**: 一种面向布局的鲁棒场景图生成方法：Robo-SGG，即基于布局的归一化与恢复方法 

---
# CM3AE: A Unified RGB Frame and Event-Voxel/-Frame Pre-training Framework 

**Title (ZH)**: CM3AE：统一的RGB帧和事件体素/帧预训练框架 

**Authors**: Wentao Wu, Xiao Wang, Chenglong Li, Bo Jiang, Jin Tang, Bin Luo, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12576)  

**Abstract**: Event cameras have attracted increasing attention in recent years due to their advantages in high dynamic range, high temporal resolution, low power consumption, and low latency. Some researchers have begun exploring pre-training directly on event data. Nevertheless, these efforts often fail to establish strong connections with RGB frames, limiting their applicability in multi-modal fusion scenarios. To address these issues, we propose a novel CM3AE pre-training framework for the RGB-Event perception. This framework accepts multi-modalities/views of data as input, including RGB images, event images, and event voxels, providing robust support for both event-based and RGB-event fusion based downstream tasks. Specifically, we design a multi-modal fusion reconstruction module that reconstructs the original image from fused multi-modal features, explicitly enhancing the model's ability to aggregate cross-modal complementary information. Additionally, we employ a multi-modal contrastive learning strategy to align cross-modal feature representations in a shared latent space, which effectively enhances the model's capability for multi-modal understanding and capturing global dependencies. We construct a large-scale dataset containing 2,535,759 RGB-Event data pairs for the pre-training. Extensive experiments on five downstream tasks fully demonstrated the effectiveness of CM3AE. Source code and pre-trained models will be released on this https URL. 

**Abstract (ZH)**: 基于RGB-事件感知的新型CM3AE预训练框架 

---
# Decision-based AI Visual Navigation for Cardiac Ultrasounds 

**Title (ZH)**: 基于决策的AI视觉导航心血管超声 

**Authors**: Andy Dimnaku, Dominic Yurk, Zhiyuan Gao, Arun Padmanabhan, Mandar Aras, Yaser Abu-Mostafa  

**Link**: [PDF](https://arxiv.org/pdf/2504.12535)  

**Abstract**: Ultrasound imaging of the heart (echocardiography) is widely used to diagnose cardiac diseases. However, obtaining an echocardiogram requires an expert sonographer and a high-quality ultrasound imaging device, which are generally only available in hospitals. Recently, AI-based navigation models and algorithms have been used to aid novice sonographers in acquiring the standardized cardiac views necessary to visualize potential disease pathologies. These navigation systems typically rely on directional guidance to predict the necessary rotation of the ultrasound probe. This paper demonstrates a novel AI navigation system that builds on a decision model for identifying the inferior vena cava (IVC) of the heart. The decision model is trained offline using cardiac ultrasound videos and employs binary classification to determine whether the IVC is present in a given ultrasound video. The underlying model integrates a novel localization algorithm that leverages the learned feature representations to annotate the spatial location of the IVC in real-time. Our model demonstrates strong localization performance on traditional high-quality hospital ultrasound videos, as well as impressive zero-shot performance on lower-quality ultrasound videos from a more affordable Butterfly iQ handheld ultrasound machine. This capability facilitates the expansion of ultrasound diagnostics beyond hospital settings. Currently, the guidance system is undergoing clinical trials and is available on the Butterfly iQ app. 

**Abstract (ZH)**: 基于AI的心脏超声成像导航系统：识别下腔静脉的新颖决策模型与应用 

---
# AdaVid: Adaptive Video-Language Pretraining 

**Title (ZH)**: AdaVid：自适应视频-语言预训练 

**Authors**: Chaitanya Patel, Juan Carlos Niebles, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2504.12513)  

**Abstract**: Contrastive video-language pretraining has demonstrated great success in learning rich and robust video representations. However, deploying such video encoders on compute-constrained edge devices remains challenging due to their high computational demands. Additionally, existing models are typically trained to process only short video clips, often limited to 4 to 64 frames. In this paper, we introduce AdaVid, a flexible architectural framework designed to learn efficient video encoders that can dynamically adapt their computational footprint based on available resources. At the heart of AdaVid is an adaptive transformer block, inspired by Matryoshka Representation Learning, which allows the model to adjust its hidden embedding dimension at inference time. We show that AdaVid-EgoVLP, trained on video-narration pairs from the large-scale Ego4D dataset, matches the performance of the standard EgoVLP on short video-language benchmarks using only half the compute, and even outperforms EgoVLP when given equal computational resources. We further explore the trade-off between frame count and compute on the challenging Diving48 classification benchmark, showing that AdaVid enables the use of more frames without exceeding computational limits. To handle longer videos, we also propose a lightweight hierarchical network that aggregates short clip features, achieving a strong balance between compute efficiency and accuracy across several long video benchmarks. 

**Abstract (ZH)**: 自适应视频编码器学习的对比视频-语言预训练在学习丰富的稳健视频表示方面取得了巨大成功。然而，由于其高计算需求，在计算资源受限的边缘设备上部署这些视频编码器仍然具有挑战性。此外，现有模型通常仅限于处理长度较短的视频片段，通常仅限于4到64帧。在本文中，我们引入了AdaVid，一个灵活的架构框架，设计用于学习高效的视频编码器，可根据可用资源动态调整其计算足迹。AdaVid的核心是一个受Matryoshka表示学习启发的自适应变压器块，在推断时允许模型调整其隐藏嵌入维度。我们证明，通过在大规模Ego4D数据集上的视频叙述对训练得到的AdaVid-EgoVLP，在使用一半计算资源的情况下，在短视频-语言基准测试上匹配标准EgoVLP的性能，并且在相同计算资源下甚至优于EgoVLP。我们还在具有挑战性的Diving48分类基准测试中探讨了帧数和计算之间的权衡，显示AdaVid使使用更多帧而不超过计算限制成为可能。为了处理更长的视频，我们还提出了一种轻量级的分层网络，它聚合短片段特征，在多个长视频基准测试中实现了计算效率和准确性的良好平衡。 

---
# WaterFlow: Learning Fast & Robust Watermarks using Stable Diffusion 

**Title (ZH)**: WaterFlow: 使用稳定扩散学习快速且稳健的水印 

**Authors**: Vinay Shukla, Prachee Sharma, Ryan Rossi, Sungchul Kim, Tong Yu, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2504.12354)  

**Abstract**: The ability to embed watermarks in images is a fundamental problem of interest for computer vision, and is exacerbated by the rapid rise of generated imagery in recent times. Current state-of-the-art techniques suffer from computational and statistical challenges such as the slow execution speed for practical deployments. In addition, other works trade off fast watermarking speeds but suffer greatly in their robustness or perceptual quality. In this work, we propose WaterFlow (WF), a fast and extremely robust approach for high fidelity visual watermarking based on a learned latent-dependent watermark. Our approach utilizes a pretrained latent diffusion model to encode an arbitrary image into a latent space and produces a learned watermark that is then planted into the Fourier Domain of the latent. The transformation is specified via invertible flow layers that enhance the expressivity of the latent space of the pre-trained model to better preserve image quality while permitting robust and tractable detection. Most notably, WaterFlow demonstrates state-of-the-art performance on general robustness and is the first method capable of effectively defending against difficult combination attacks. We validate our findings on three widely used real and generated datasets: MS-COCO, DiffusionDB, and WikiArt. 

**Abstract (ZH)**: 图像中嵌入水印的能力是计算机视觉中的一个基本问题，近年来生成图像的迅速增长加剧了这一问题。当前最先进的技术面临着诸如实用部署时执行速度慢等计算和统计挑战。此外，其他工作虽然实现了快速的水印速度，但在鲁棒性和感知质量方面却遭受了极大的损失。在本文中，我们提出了一种名为WaterFlow（WF）的快速且极其鲁棒的高保真视觉水印方法，基于学习到的潜在依赖水印。该方法利用预训练的潜在扩散模型将任意图像编码到潜在空间中，并生成一个学习到的水印，然后将其植入到潜在的傅里叶域中。变换通过可逆流层指定，以增强预训练模型的潜在空间的表达能力，更好地保持图像质量的同时允许鲁棒且易于实现的检测。最值得注意的是，WaterFlow在通用鲁棒性方面展示了最先进的性能，并且是第一个能够有效防御复杂组合攻击的方法。我们在三个广泛使用的现实和生成数据集上验证了我们的发现：MS-COCO、DiffusionDB和WikiArt。 

---
# Deep Generative Model-Based Generation of Synthetic Individual-Specific Brain MRI Segmentations 

**Title (ZH)**: 基于深度生成模型的合成个体特定脑MRI分割生成 

**Authors**: Ruijie Wang, Luca Rossetto, Susan Mérillat, Christina Röcke, Mike Martin, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2504.12352)  

**Abstract**: To the best of our knowledge, all existing methods that can generate synthetic brain magnetic resonance imaging (MRI) scans for a specific individual require detailed structural or volumetric information about the individual's brain. However, such brain information is often scarce, expensive, and difficult to obtain. In this paper, we propose the first approach capable of generating synthetic brain MRI segmentations -- specifically, 3D white matter (WM), gray matter (GM), and cerebrospinal fluid (CSF) segmentations -- for individuals using their easily obtainable and often readily available demographic, interview, and cognitive test information. Our approach features a novel deep generative model, CSegSynth, which outperforms existing prominent generative models, including conditional variational autoencoder (C-VAE), conditional generative adversarial network (C-GAN), and conditional latent diffusion model (C-LDM). We demonstrate the high quality of our synthetic segmentations through extensive evaluations. Also, in assessing the effectiveness of the individual-specific generation, we achieve superior volume prediction, with Pearson correlation coefficients reaching 0.80, 0.82, and 0.70 between the ground-truth WM, GM, and CSF volumes of test individuals and those volumes predicted based on generated individual-specific segmentations, respectively. 

**Abstract (ZH)**: 已知的能够为特定个体生成合成脑磁共振成像（MRI）扫描的方法都需要该个体详细的大脑结构或容积信息。然而，此类大脑信息往往稀缺、昂贵且难以获得。在本文中，我们提出了第一个能够使用个体可轻易获得且通常容易获取的 Demographic、访谈和认知测试信息来生成合成脑 MRI 分割的方法——特别是生成 3D 白质 (WM)、灰质 (GM) 和脑脊液 (CSF) 分割。我们的方法特点是一个新颖的深度生成模型 CSegSynth，其性能优于现有的主流生成模型，包括条件变分自编码器（C-VAE）、条件生成对抗网络（C-GAN）和条件潜在扩散模型（C-LDM）。通过广泛的评估证明了我们合成分割的质量。此外，在评估个体特定生成的有效性时，我们实现了卓越的容积预测， pearson 相关系数分别达到 0.80、0.82 和 0.70，对应于测试个体的真实 WM、GM 和 CSF 容积与基于生成的个体特定分割预测的容积之间的相关性。 

---
# Data Metabolism: An Efficient Data Design Schema For Vision Language Model 

**Title (ZH)**: 数据代谢：一种高效的视觉语言模型数据设计方案 

**Authors**: Jingyuan Zhang, Hongzhi Zhang, Zhou Haonan, Chenxi Sun, Xingguang ji, Jiakang Wang, Fanheng Kong, Yahui Liu, Qi Wang, Fuzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12316)  

**Abstract**: Data curation plays a crucial role in training powerful Visual Language Models (VLMs). In this work, we introduce the concept of Data Metabolism and present our data-centric framework to build VLMs throughout the development lifecycle. Starting from a standard model architecture, we discuss and provide insights into two crucial development steps: data curation and iteration, forming a closed-loop system that continuously improves model performance. We show a detailed codebook on how to process existing massive datasets and build user-specific data flywheel. As a demonstration, we release a VLM, named Capybara-VL, which excels in typical multimodal tasks (e.g. , visual question answering, scientific reasoning, and text-rich tasks). Despite its relatively compact size, Capybara-VL surpasses several open-source models that are up to 10 times larger in size. Moreover, it achieves results that are on par with those of several leading proprietary models, demonstrating its remarkable competitiveness. These results highlight the power of our data-centric framework and the potential of training smaller and more efficient VLMs. 

**Abstract (ZH)**: 数据治理在训练强大视觉语言模型中的作用至关重要。在这项工作中，我们引入了数据新陈代谢的概念，并提出了一种以数据为中心的框架，贯穿视觉语言模型开发的整个生命周期。从标准模型架构出发，我们讨论并提供了数据治理和迭代两个关键步骤的见解，形成一个闭环系统，持续提高模型性能。我们详细介绍了如何处理现有大规模数据集并构建用户特定的数据飞轮。作为演示，我们发布了名为Capybara-VL的视觉语言模型，该模型在典型的多模态任务（如视觉问答、科学推理和图文任务）上表现出色。尽管其相对较小，Capybara-VL仍超越了多个开源模型，这些模型的规模是其的10倍以上。此外，它还达到了与多个领先私有模型相当的结果，展示了其显著的竞争优势。这些结果突显了我们以数据为中心的框架的力量以及训练更小、更高效的视觉语言模型的潜力。 

---
