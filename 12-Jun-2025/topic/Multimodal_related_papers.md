# From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models 

**Title (ZH)**: 从意图到执行：探究视觉-语言-行动模型的泛化边界 

**Authors**: Irving Fang, Juexiao Zhang, Shengbang Tong, Chen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.09930)  

**Abstract**: One promise that Vision-Language-Action (VLA) models hold over traditional imitation learning for robotics is to leverage the broad generalization capabilities of large Vision-Language Models (VLMs) to produce versatile, "generalist" robot policies. However, current evaluations of VLAs remain insufficient. Traditional imitation learning benchmarks are unsuitable due to the lack of language instructions. Emerging benchmarks for VLAs that incorporate language often come with limited evaluation tasks and do not intend to investigate how much VLM pretraining truly contributes to the generalization capabilities of the downstream robotic policy. Meanwhile, much research relies on real-world robot setups designed in isolation by different institutions, which creates a barrier for reproducibility and accessibility. To address this gap, we introduce a unified probing suite of 50 simulation-based tasks across 10 subcategories spanning language instruction, vision, and objects. We systematically evaluate several state-of-the-art VLA architectures on this suite to understand their generalization capability. Our results show that while VLM backbones endow VLAs with robust perceptual understanding and high level planning, which we refer to as good intentions, this does not reliably translate into precise motor execution: when faced with out-of-distribution observations, policies often exhibit coherent intentions, but falter in action execution. Moreover, finetuning on action data can erode the original VLM's generalist reasoning abilities. We release our task suite and evaluation code to serve as a standardized benchmark for future VLAs and to drive research on closing the perception-to-action gap. More information, including the source code, can be found at this https URL 

**Abstract (ZH)**: Vision-Language-Action模型在机器人领域超越传统模仿学习的 promise在于利用大体量视觉-语言模型的广泛泛化能力生成多用途的“通才”机器人策略，但当前对Vision-Language-Action (VLA)模型的评估仍显不足。传统的模仿学习基准由于缺乏语言指令而不适用。新兴的VLA基准虽然包含了语言，但评估任务有限，并未深入探究VLM预训练对下游机器人策略泛化能力的实际贡献。同时，许多研究依赖于不同机构独立设计的现实世界机器人设置，这造成了可重复性和可访问性的障碍。为解决这一问题，我们引入了一个统一的探针套件，包括涵盖语言指令、视觉和物体在内的10个子类别的50项仿真任务。我们系统性地评估了几种最先进的VLA架构，以了解其泛化能力。结果显示，尽管VLM骨干网络赋予了VLA模型强大的感知理解和高层规划能力，即所谓的“好意图”，但这并不一定能可靠地转化为精准的行动执行：当面对分布外的观察时，策略常常表现出一致的意图，但在行动执行方面却失败了。此外，通过行动数据进行微调可能会削弱原始VLM的通用推理能力。我们发布了该任务套件和评估代码，以作为未来VLA的标准化基准，并推动缩小感知到行动差距的研究。更多信息，包括源代码，可在以下链接获取。 

---
# DCIRNet: Depth Completion with Iterative Refinement for Dexterous Grasping of Transparent and Reflective Objects 

**Title (ZH)**: DCIRNet：用于透明和反射物体灵活抓取的迭代完善深度完成 

**Authors**: Guanghu Xie, Zhiduo Jiang, Yonglong Zhang, Yang Liu, Zongwu Xie, Baoshi Cao, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09491)  

**Abstract**: Transparent and reflective objects in everyday environments pose significant challenges for depth sensors due to their unique visual properties, such as specular reflections and light transmission. These characteristics often lead to incomplete or inaccurate depth estimation, which severely impacts downstream geometry-based vision tasks, including object recognition, scene reconstruction, and robotic manipulation. To address the issue of missing depth information in transparent and reflective objects, we propose DCIRNet, a novel multimodal depth completion network that effectively integrates RGB images and depth maps to enhance depth estimation quality. Our approach incorporates an innovative multimodal feature fusion module designed to extract complementary information between RGB images and incomplete depth maps. Furthermore, we introduce a multi-stage supervision and depth refinement strategy that progressively improves depth completion and effectively mitigates the issue of blurred object boundaries. We integrate our depth completion model into dexterous grasping frameworks and achieve a $44\%$ improvement in the grasp success rate for transparent and reflective objects. We conduct extensive experiments on public datasets, where DCIRNet demonstrates superior performance. The experimental results validate the effectiveness of our approach and confirm its strong generalization capability across various transparent and reflective objects. 

**Abstract (ZH)**: 透明和反射物体在日常生活环境中的深度传感中由于其独特的视觉特性（如镜面反射和透光性）造成了重大挑战，这常常导致不完整或不准确的深度估计，严重影响基于几何的下游视觉任务，包括物体识别、场景重建和机器人操作。为了应对透明和反射物体中缺失的深度信息问题，我们提出了一种新颖的多模态深度完成网络DCIRNet，该网络有效整合了RGB图像和深度图以提升深度估计质量。我们的方法包含一种创新的多模态特征融合模块，用于提取RGB图像和不完整深度图之间的互补信息。此外，我们还提出了一种多阶段监督和深度细化策略，逐步提高深度完成质量并有效缓解物体边界模糊的问题。我们将深度完成模型集成到灵巧抓取框架中，对于透明和反射物体实现了44%的抓取成功率提升。我们在公共数据集上进行了广泛的实验，结果表明DCIRNet具有优越性能，并证实了该方法在不同透明和反射物体上的强泛化能力。 

---
# Ming-Omni: A Unified Multimodal Model for Perception and Generation 

**Title (ZH)**: 明-全知：统一的多模态感知与生成模型 

**Authors**: Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, Furong Xu, GuangMing Yao, Jun Zhou, Jingdong Chen, Jianxin Sun, Jiajia Liu, Jianjiang Zhu, Jun Peng, Kaixiang Ji, Kaiyou Song, Kaimeng Ren, Libin Wang, Lixiang Ru, Lele Xie, Longhua Tan, Lyuxin Xue, Lan Wang, Mochen Bai, Ning Gao, Pei Chen, Qingpei Guo, Qinglong Zhang, Qiang Xu, Rui Liu, Ruijie Xiong, Sirui Gao, Tinghao Liu, Taisong Li, Weilong Chai, Xinyu Xiao, Xiaomei Wang, Xiaoxue Chen, Xiao Lu, Xiaoyu Li, Xingning Dong, Xuzheng Yu, Yi Yuan, Yuting Gao, Yunxiao Sun, Yipeng Chen, Yifei Wu, Yongjie Lyu, Ziping Ma, Zipeng Feng, Zhijiang Fang, Zhihao Qiu, Ziyuan Huang, Zhengyu He  

**Link**: [PDF](https://arxiv.org/pdf/2506.09344)  

**Abstract**: We propose Ming-Omni, a unified multimodal model capable of processing images, text, audio, and video, while demonstrating strong proficiency in both speech and image generation. Ming-Omni employs dedicated encoders to extract tokens from different modalities, which are then processed by Ling, an MoE architecture equipped with newly proposed modality-specific routers. This design enables a single model to efficiently process and fuse multimodal inputs within a unified framework, thereby facilitating diverse tasks without requiring separate models, task-specific fine-tuning, or structural redesign. Importantly, Ming-Omni extends beyond conventional multimodal models by supporting audio and image generation. This is achieved through the integration of an advanced audio decoder for natural-sounding speech and Ming-Lite-Uni for high-quality image generation, which also allow the model to engage in context-aware chatting, perform text-to-speech conversion, and conduct versatile image editing. Our experimental results showcase Ming-Omni offers a powerful solution for unified perception and generation across all modalities. Notably, our proposed Ming-Omni is the first open-source model we are aware of to match GPT-4o in modality support, and we release all code and model weights to encourage further research and development in the community. 

**Abstract (ZH)**: 我们提出Ming-Omni，一种统一的多模态模型，能够处理图像、文本、音频和视频，并在语音和图像生成方面展现出强大的能力。Ming-Omni采用专门的编码器从不同模态中提取令牌，这些令牌随后由装备有新提出的模态特定路由器的MoE架构Ling进行处理。这种设计使得单一模型能够在统一框架内高效地处理和融合多模态输入，从而实现在无需单独模型、任务特定微调或结构重设计的情况下完成多种任务。更重要的是，Ming-Omni超越了传统的多模态模型，支持音频和图像生成。这通过集成先进的音频解码器实现自然声音语音，并结合Ming-Lite-Uni进行高质量图像生成，使模型能够进行上下文感知聊天、文本转语音转换和多用途图像编辑。我们的实验结果展示了Ming-Omni提供了跨所有模态统一感知和生成的强大解决方案。值得注意的是，我们提出的Ming-Omni是我们所知的第一个开源模型，能够在模态支持方面与GPT-4o相媲美，我们发布了全部代码和模型权重，以鼓励社区进一步的研究和开发。 

---
# InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions 

**Title (ZH)**: InterActHuman：布局对齐音频条件驱动的多概念人体动画生成 

**Authors**: Zhenzhi Wang, Jiaqi Yang, Jianwen Jiang, Chao Liang, Gaojie Lin, Zerong Zheng, Ceyuan Yang, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09984)  

**Abstract**: End-to-end human animation with rich multi-modal conditions, e.g., text, image and audio has achieved remarkable advancements in recent years. However, most existing methods could only animate a single subject and inject conditions in a global manner, ignoring scenarios that multiple concepts could appears in the same video with rich human-human interactions and human-object interactions. Such global assumption prevents precise and per-identity control of multiple concepts including humans and objects, therefore hinders applications. In this work, we discard the single-entity assumption and introduce a novel framework that enforces strong, region-specific binding of conditions from modalities to each identity's spatiotemporal footprint. Given reference images of multiple concepts, our method could automatically infer layout information by leveraging a mask predictor to match appearance cues between the denoised video and each reference appearance. Furthermore, we inject local audio condition into its corresponding region to ensure layout-aligned modality matching in a iterative manner. This design enables the high-quality generation of controllable multi-concept human-centric videos. Empirical results and ablation studies validate the effectiveness of our explicit layout control for multi-modal conditions compared to implicit counterparts and other existing methods. 

**Abstract (ZH)**: 端到端多模态条件下的丰富人体动画：摆脱单一主体假设，实现精确的身份特定控制 

---
# Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing 

**Title (ZH)**: 在交织思考与视觉绘制中强化视觉语言模型的空间推理能力 

**Authors**: Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09965)  

**Abstract**: As textual reasoning with large language models (LLMs) has advanced significantly, there has been growing interest in enhancing the multimodal reasoning capabilities of large vision-language models (LVLMs). However, existing methods primarily approach multimodal reasoning in a straightforward, text-centric manner, where both reasoning and answer derivation are conducted purely through text, with the only difference being the presence of multimodal input. As a result, these methods often encounter fundamental limitations in spatial reasoning tasks that demand precise geometric understanding and continuous spatial tracking-capabilities that humans achieve through mental visualization and manipulation. To address the limitations, we propose drawing to reason in space, a novel paradigm that enables LVLMs to reason through elementary drawing operations in the visual space. By equipping models with basic drawing operations, including annotating bounding boxes and drawing auxiliary lines, we empower them to express and analyze spatial relationships through direct visual manipulation, meanwhile avoiding the performance ceiling imposed by specialized perception tools in previous tool-integrated reasoning approaches. To cultivate this capability, we develop a three-stage training framework: cold-start training with synthetic data to establish basic drawing abilities, reflective rejection sampling to enhance self-reflection behaviors, and reinforcement learning to directly optimize for target rewards. Extensive experiments demonstrate that our model, named VILASR, consistently outperforms existing methods across diverse spatial reasoning benchmarks, involving maze navigation, static spatial reasoning, video-based reasoning, and multi-view-based reasoning tasks, with an average improvement of 18.4%. 

**Abstract (ZH)**: 基于空间绘图的视觉语言模型空间推理新范式 

---
# 3D-Aware Vision-Language Models Fine-Tuning with Geometric Distillation 

**Title (ZH)**: 三维意识视觉-语言模型几何蒸馏 fine-tuning 

**Authors**: Seonho Lee, Jiho Choi, Inha Kang, Jiwook Kim, Junsung Park, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09883)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable performance on diverse visual and linguistic tasks, yet they remain fundamentally limited in their understanding of 3D spatial structures. We propose Geometric Distillation, a lightweight, annotation-free fine-tuning framework that injects human-inspired geometric cues into pretrained VLMs without modifying their architecture. By distilling (1) sparse correspondences, (2) relative depth relations, and (3) dense cost volumes from off-the-shelf 3D foundation models (e.g., MASt3R, VGGT), our method shapes representations to be geometry-aware while remaining compatible with natural image-text inputs. Through extensive evaluations on 3D vision-language reasoning and 3D perception benchmarks, our method consistently outperforms prior approaches, achieving improved 3D spatial reasoning with significantly lower computational cost. Our work demonstrates a scalable and efficient path to bridge 2D-trained VLMs with 3D understanding, opening up wider use in spatially grounded multimodal tasks. 

**Abstract (ZH)**: 几何蒸馏：一种轻量级、无注释的细调框架，将人类启发的几何线索注入预训练的视觉-语言模型以增强三维空间理解 

---
# Vision Matters: Simple Visual Perturbations Can Boost Multimodal Math Reasoning 

**Title (ZH)**: 视觉很重要：简单的视觉扰动可以提升多模态数学推理能力 

**Authors**: Yuting Li, Lai Wei, Kaipeng Zheng, Jingyuan Huang, Linghe Kong, Lichao Sun, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09736)  

**Abstract**: Despite the rapid progress of multimodal large language models (MLLMs), they have largely overlooked the importance of visual processing. In a simple yet revealing experiment, we interestingly find that language-only models, when provided with image captions, can achieve comparable or even better performance than MLLMs that consume raw visual inputs. This suggests that current MLLMs may generate accurate visual descriptions but fail to effectively integrate them during reasoning. Motivated by this, we propose a simple visual perturbation framework that enhances perceptual robustness without requiring algorithmic modifications or additional training data. Our approach introduces three targeted perturbations: distractor concatenation, dominance-preserving mixup, and random rotation, that can be easily integrated into existing post-training pipelines including SFT, DPO, and GRPO. Through extensive experiments across multiple datasets, we demonstrate consistent improvements in mathematical reasoning performance, with gains comparable to those achieved through algorithmic changes. Additionally, we achieve competitive performance among open-source 7B RL-tuned models by training Qwen2.5-VL-7B with visual perturbation. Through comprehensive ablation studies, we analyze the effectiveness of different perturbation strategies, revealing that each perturbation type contributes uniquely to different aspects of visual reasoning. Our findings highlight the critical role of visual perturbation in multimodal mathematical reasoning: better reasoning begins with better seeing. Our code is available at this https URL. 

**Abstract (ZH)**: 尽管多模态大型语言模型（MLLMs）取得了快速进展，但它们在视觉处理的重要性上还缺乏足够的关注。在一项简单而富有启发性的实验中，我们发现，仅依靠语言的语言模型，在获得图像描述后，可以实现与消费原始视觉输入的MLLM相当甚至更优的性能。这表明当前的MLLM可能能够生成准确的视觉描述，但在推理过程中未能有效整合它们。受此启发，我们提出了一种简单的视觉扰动框架，该框架可以增强感知鲁棒性，而无需进行算法修改或额外的数据训练。我们的方法引入了三种有针对性的扰动：干扰项连接、保持主导性的混合、以及随机旋转，这些扰动可以轻松集成到现有的后训练管道中，包括SFT、DPO和GRPO。通过在多个数据集上的广泛实验，我们展示了在数学推理性能上的一致改进，这些改进与通过算法变化所达到的增益相当。此外，通过视觉扰动训练Qwen2.5-VL-7B，我们实现了开源7B RL调优模型的竞争力。通过全面的消融研究，我们分析了不同扰动策略的有效性，揭示了每种扰动类型对视觉推理的不同方面具有独特的贡献。我们的研究结果强调了在多模态数学推理中视觉扰动的关键作用：更好的推理始于更好的视觉能力。我们的代码可在以下链接获取：this https URL 

---
# HSENet: Hybrid Spatial Encoding Network for 3D Medical Vision-Language Understanding 

**Title (ZH)**: HSENet：用于3D医疗视觉-语言理解的混合空间编码网络 

**Authors**: Yanzhao Shi, Xiaodan Zhang, Junzhong Ji, Haoning Jiang, Chengxin Zheng, Yinong Wang, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09634)  

**Abstract**: Automated 3D CT diagnosis empowers clinicians to make timely, evidence-based decisions by enhancing diagnostic accuracy and workflow efficiency. While multimodal large language models (MLLMs) exhibit promising performance in visual-language understanding, existing methods mainly focus on 2D medical images, which fundamentally limits their ability to capture complex 3D anatomical structures. This limitation often leads to misinterpretation of subtle pathologies and causes diagnostic hallucinations. In this paper, we present Hybrid Spatial Encoding Network (HSENet), a framework that exploits enriched 3D medical visual cues by effective visual perception and projection for accurate and robust vision-language understanding. Specifically, HSENet employs dual-3D vision encoders to perceive both global volumetric contexts and fine-grained anatomical details, which are pre-trained by dual-stage alignment with diagnostic reports. Furthermore, we propose Spatial Packer, an efficient multimodal projector that condenses high-resolution 3D spatial regions into a compact set of informative visual tokens via centroid-based compression. By assigning spatial packers with dual-3D vision encoders, HSENet can seamlessly perceive and transfer hybrid visual representations to LLM's semantic space, facilitating accurate diagnostic text generation. Experimental results demonstrate that our method achieves state-of-the-art performance in 3D language-visual retrieval (39.85% of R@100, +5.96% gain), 3D medical report generation (24.01% of BLEU-4, +8.01% gain), and 3D visual question answering (73.60% of Major Class Accuracy, +1.99% gain), confirming its effectiveness. Our code is available at this https URL. 

**Abstract (ZH)**: 自动化3D CT诊断赋能临床医生通过增强诊断准确性和工作流程效率及时做出基于证据的决策。现有方法主要集中在2D医学图像上，这从根本上限制了其捕捉复杂3D解剖结构的能力。受限于此，往往会误解细微的病理特征，导致诊断幻觉。本文提出了结合空间编码网络（HSENet），该框架通过有效的视觉感知和投影利用丰富的3D医学视觉线索，以实现准确和鲁棒的视觉语言理解。具体而言，HSENet 使用双3D视觉编码器感知全局体素上下文和精细的解剖细节，这些细节通过双重阶段对齐预训练诊断报告。此外，我们提出了空间打包器（Spatial Packer），这是一种高效多模态投影器，通过基于质心的压缩将高分辨率3D空间区域凝缩为一组信息丰富的视觉标记。通过赋予空间打包器双3D视觉编码器，HSENet 可以无缝地感知和传递混合视觉表示到大语言模型（LLM）的语义空间，促进准确的诊断文本生成。实验结果证明，我们的方法在3D语言视觉检索（39.85%的R@100，+5.96%的提高）、3D医学报告生成（24.01%的BLEU-4，+8.01%的提高）和3D视觉问答（73.60%的主要类别准确度，+1.99%的提高）中达到了最先进的性能，证实了其有效性。我们的代码可在以下链接获取。 

---
# Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models 

**Title (ZH)**: Athena: 以数据高效的过程奖励模型增强多模态推理 

**Authors**: Shuai Wang, Zhenhua Liu, Jiaheng Wei, Xuanwu Yin, Dong Li, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.09532)  

**Abstract**: We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks. 

**Abstract (ZH)**: Athena-PRM: 多模态过程奖励模型及其在复杂推理问题评估中的应用 

---
# Revisit What You See: Disclose Language Prior in Vision Tokens for Efficient Guided Decoding of LVLMs 

**Title (ZH)**: 重访你所看见的：在视觉标记中披露语言先验以实现高效的LVLM引导解码 

**Authors**: Beomsik Cho, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09522)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across various multimodal tasks by integrating visual perception with language understanding. However, conventional decoding strategies of LVLMs often fail to successfully utilize visual information, leading to visually ungrounded responses. While various approaches have been proposed to address this limitation, they typically require additional training, multi-step inference procedures, or external model dependencies. This paper introduces ReVisiT, a simple yet effective decoding method that references vision tokens to guide the text generation process in LVLMs. Our approach leverages the semantic information embedded within vision tokens by projecting them into the text token distribution space, and dynamically selecting the most relevant vision token at each decoding step through constrained divergence minimization. This selected vision token is then used to refine the output distribution to better incorporate visual semantics. Experiments on three LVLM hallucination benchmarks with two recent LVLMs demonstrate that ReVisiT consistently enhances visual grounding with minimal computational overhead. Moreover, our method achieves competitive or superior results relative to state-of-the-art baselines while reducing computational costs for up to $2\times$. 

**Abstract (ZH)**: 大规模 vision-language 模型 (LVLMs) 在多模态任务中通过结合视觉感知和语言理解展现了卓越的表现。然而，LVLMs 传统的解码策略往往无法有效利用视觉信息，导致生成的回答与视觉内容脱节。尽管提出了多种方法来解决这一限制，它们通常需要额外的训练、多步推理流程或外部模型依赖。本文引入了 ReVisiT，这是一种简单而有效的解码方法，通过引用视觉标记来指导 LVLMs 的文本生成过程。我们的方法通过将视觉标记投影到文本标记分布空间并动态选择与当前解码步骤最相关的视觉标记来利用嵌入在其内的语义信息，从而通过受限的偏差最小化进行选择。选择的视觉标记随后用于细化输出分布，使其更好地融入视觉语义。在两个最新 LVLMs 上对三个 LVLM 幻觉基准的实验表明，ReVisiT 以最小的计算开销一致地提升了视觉定位能力。此外，与最先进的基线方法相比，我们的方法在某些情况下实现了可竞争或更优的结果，同时将计算成本降低了最多 $2\times$。 

---
# A High-Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation 

**Title (ZH)**: 高质量数据集与可靠的 interleaved 图像-文本 生成评估 

**Authors**: Yukang Feng, Jianwen Sun, Chuanhao Li, Zizhen Li, Jiaxin Ai, Fanrui Zhang, Yifan Chang, Sizhuo Zhou, Shenglin Zhang, Yu Dai, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09427)  

**Abstract**: Recent advancements in Large Multimodal Models (LMMs) have significantly improved multimodal understanding and generation. However, these models still struggle to generate tightly interleaved image-text outputs, primarily due to the limited scale, quality and instructional richness of current training datasets. To address this, we introduce InterSyn, a large-scale multimodal dataset constructed using our Self-Evaluation with Iterative Refinement (SEIR) method. InterSyn features multi-turn, instruction-driven dialogues with tightly interleaved imagetext responses, providing rich object diversity and rigorous automated quality refinement, making it well-suited for training next-generation instruction-following LMMs. Furthermore, to address the lack of reliable evaluation tools capable of assessing interleaved multimodal outputs, we introduce SynJudge, an automatic evaluation model designed to quantitatively assess multimodal outputs along four dimensions: text content, image content, image quality, and image-text synergy.
Experimental studies show that the SEIR method leads to substantially higher dataset quality compared to an otherwise identical process without refinement.
Moreover, LMMs trained on InterSyn achieve uniform performance gains across all evaluation metrics, confirming InterSyn's utility for advancing multimodal systems. 

**Abstract (ZH)**: 近期大型多模态模型（LMMs）的进展显著提高了多模态的理解和生成能力，但这些模型仍然难以生成紧密交织的图像-文本输出，主要原因是当前训练数据集的规模、质量和指导丰富性有限。为解决这一问题，我们引入了InterSyn，这是一种使用我们自评价与迭代细化（SEIR）方法构建的大规模多模态数据集。InterSyn 包含多轮、指令驱动的对话，具有紧密交织的图像-文本响应，提供丰富的对象多样性并进行严格的自动化质量 refinement，使其非常适合训练下一代遵循指令的LMMs。此外，为了解决缺乏可靠的评估工具来评估交织的多模态输出的问题，我们引入了SynJudge，这是一种自动评估模型，旨在从文本内容、图像内容、图像质量和图像-文本协同作用四个维度定量评估多模态输出。实验研究显示，SEIR方法导致了显著更高的数据集质量。此外，使用InterSyn训练的LMMs在整个评估指标上均实现了均匀的性能提升，进一步证明了InterSyn对推动多模态系统发展的实用性。 

---
# FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation 

**Title (ZH)**: FlagEvalMM: 一种灵活的全面多模态模型评估框架 

**Authors**: Zheqi He, Yesheng Liu, Jing-shu Zheng, Xuejing Li, Richeng Xuan, Jin-Ge Yao, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09081)  

**Abstract**: We present FlagEvalMM, an open-source evaluation framework designed to comprehensively assess multimodal models across a diverse range of vision-language understanding and generation tasks, such as visual question answering, text-to-image/video generation, and image-text retrieval. We decouple model inference from evaluation through an independent evaluation service, thus enabling flexible resource allocation and seamless integration of new tasks and models. Moreover, FlagEvalMM utilizes advanced inference acceleration tools (e.g., vLLM, SGLang) and asynchronous data loading to significantly enhance evaluation efficiency. Extensive experiments show that FlagEvalMM offers accurate and efficient insights into model strengths and limitations, making it a valuable tool for advancing multimodal research. The framework is publicly accessible athttps://github.com/flageval-baai/FlagEvalMM. 

**Abstract (ZH)**: 我们介绍了一个开源评价框架FlagEvalMM，该框架旨在全面评估涵盖视觉语言理解和生成任务（如视觉问答、文本生成图像/视频以及图像-文本检索）的多模态模型。通过独立的评价服务将模型推理与评价分离，从而实现灵活的资源分配和新任务与模型的无缝集成。此外，FlagEvalMM 利用先进的推理加速工具（如 vLLM、SGLang）和异步数据加载显著提升评价效率。大量实验表明，FlagEvalMM 提供了准确而高效的模型优势与局限性洞察，成为推动多模态研究进展的重要工具。该框架在 https://github.com/flageval-baai/FlagEvalMM 公开可用。 

---
# VersaVid-R1: A Versatile Video Understanding and Reasoning Model from Question Answering to Captioning Tasks 

**Title (ZH)**: VersaVid-R1：从问答到字幕生成的通用视频理解与推理模型 

**Authors**: Xinlong Chen, Yuanxing Zhang, Yushuo Guan, Bohan Zeng, Yang Shi, Sihan Yang, Pengfei Wan, Qiang Liu, Liang Wang, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09079)  

**Abstract**: Recent advancements in multimodal large language models have successfully extended the Reason-Then-Respond paradigm to image-based reasoning, yet video-based reasoning remains an underdeveloped frontier, primarily due to the scarcity of high-quality reasoning-oriented data and effective training methodologies. To bridge this gap, we introduce DarkEventInfer and MixVidQA, two novel datasets specifically designed to stimulate the model's advanced video understanding and reasoning abilities. DarkEventinfer presents videos with masked event segments, requiring models to infer the obscured content based on contextual video cues. MixVidQA, on the other hand, presents interleaved video sequences composed of two distinct clips, challenging models to isolate and reason about one while disregarding the other. Leveraging these carefully curated training samples together with reinforcement learning guided by diverse reward functions, we develop VersaVid-R1, the first versatile video understanding and reasoning model under the Reason-Then-Respond paradigm capable of handling multiple-choice and open-ended question answering, as well as video captioning tasks. Extensive experiments demonstrate that VersaVid-R1 significantly outperforms existing models across a broad spectrum of benchmarks, covering video general understanding, cognitive reasoning, and captioning tasks. 

**Abstract (ZH)**: 近期多模态大型语言模型的进展已成功将Reason-Then-Respond范式扩展到基于图像的推理，但基于视频的推理仍然是一个欠开发的前沿领域，主要由于高质量的推理导向数据和有效的训练方法的匮乏。为了弥合这一差距，我们介绍了DarkEventInfer和MixVidQA两个新的数据集，旨在刺激模型的高级视频理解与推理能力。DarkEventInfer展示了被遮蔽事件段的视频，要求模型根据上下文视频线索推断被遮蔽的内容。MixVidQA则展示了交织的视频序列，由两段不同的片段组成，挑战模型在忽略一段的同时对另一段进行分离和推理。借助这些精心策划的训练样本以及由多种奖励函数引导的强化学习，我们开发了VersaVid-R1，这是首个能够在Reason-Then-Respond范式下处理多项选择和开放式问答以及视频字幕任务的多功能视频理解与推理模型。广泛的实验表明，VersaVid-R1在涵盖视频一般理解、认知推理和字幕任务的一系列基准测试中显著优于现有模型。 

---
# Segment Any Architectural Facades (SAAF):An automatic segmentation model for building facades, walls and windows based on multimodal semantics guidance 

**Title (ZH)**: 基于多模态语义引导的建筑 façade、墙体和窗户自动分割模型：任意分割建筑 façade（SAAF） 

**Authors**: Peilin Li, Jun Yin, Jing Zhong, Ran Luo, Pengyu Zeng, Miao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09071)  

**Abstract**: In the context of the digital development of architecture, the automatic segmentation of walls and windows is a key step in improving the efficiency of building information models and computer-aided design. This study proposes an automatic segmentation model for building facade walls and windows based on multimodal semantic guidance, called Segment Any Architectural Facades (SAAF). First, SAAF has a multimodal semantic collaborative feature extraction mechanism. By combining natural language processing technology, it can fuse the semantic information in text descriptions with image features, enhancing the semantic understanding of building facade components. Second, we developed an end-to-end training framework that enables the model to autonomously learn the mapping relationship from text descriptions to image segmentation, reducing the influence of manual intervention on the segmentation results and improving the automation and robustness of the model. Finally, we conducted extensive experiments on multiple facade datasets. The segmentation results of SAAF outperformed existing methods in the mIoU metric, indicating that the SAAF model can maintain high-precision segmentation ability when faced with diverse datasets. Our model has made certain progress in improving the accuracy and generalization ability of the wall and window segmentation task. It is expected to provide a reference for the development of architectural computer vision technology and also explore new ideas and technical paths for the application of multimodal learning in the architectural field. 

**Abstract (ZH)**: 基于多模态语义指导的建筑立面墙体和窗口自动分割模型（SAAF） 

---
