# Towards Fusing Point Cloud and Visual Representations for Imitation Learning 

**Title (ZH)**: 面向点云和视觉表示融合的imitation learning研究 

**Authors**: Atalay Donat, Xiaogang Jia, Xi Huang, Aleksandar Taranovic, Denis Blessing, Ge Li, Hongyi Zhou, Hanyi Zhang, Rudolf Lioutikov, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2502.12320)  

**Abstract**: Learning for manipulation requires using policies that have access to rich sensory information such as point clouds or RGB images. Point clouds efficiently capture geometric structures, making them essential for manipulation tasks in imitation learning. In contrast, RGB images provide rich texture and semantic information that can be crucial for certain tasks. Existing approaches for fusing both modalities assign 2D image features to point clouds. However, such approaches often lose global contextual information from the original images. In this work, we propose a novel imitation learning method that effectively combines the strengths of both point cloud and RGB modalities. Our method conditions the point-cloud encoder on global and local image tokens using adaptive layer norm conditioning, leveraging the beneficial properties of both modalities. Through extensive experiments on the challenging RoboCasa benchmark, we demonstrate the limitations of relying on either modality alone and show that our method achieves state-of-the-art performance across all tasks. 

**Abstract (ZH)**: 学习用于操作任务需要使用能够访问丰富感官信息（如点云或RGB图像）的策略。点云有效地捕获几何结构，使其成为模仿学习中操作任务的关键。相比之下，RGB图像提供了丰富的纹理和语义信息，对于某些任务至关重要。现有的方法通过将2D图像特征分配给点云来融合这两种模态，但这些方法往往会从原始图像中丢失全局上下文信息。在本工作中，我们提出了一种新颖的模仿学习方法，该方法有效地结合了点云和RGB模态的优势。我们的方法通过自适应层规范条件使得点云编码器依赖于全局和局部图像标记，利用这两种模态的有益特性。通过在具有挑战性的RoboCasa基准上的广泛实验，我们展示了仅依赖任一模态的局限性，并证明了我们的方法在所有任务中都达到了最先进的性能。 

---
# AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors 

**Title (ZH)**: AnyTouch: 学习多传感器视触觉统一静态-动态表示 

**Authors**: Ruoxuan Feng, Jiangyu Hu, Wenke Xia, Tianci Gao, Ao Shen, Yuhao Sun, Bin Fang, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12191)  

**Abstract**: Visuo-tactile sensors aim to emulate human tactile perception, enabling robots to precisely understand and manipulate objects. Over time, numerous meticulously designed visuo-tactile sensors have been integrated into robotic systems, aiding in completing various tasks. However, the distinct data characteristics of these low-standardized visuo-tactile sensors hinder the establishment of a powerful tactile perception system. We consider that the key to addressing this issue lies in learning unified multi-sensor representations, thereby integrating the sensors and promoting tactile knowledge transfer between them. To achieve unified representation of this nature, we introduce TacQuad, an aligned multi-modal multi-sensor tactile dataset from four different visuo-tactile sensors, which enables the explicit integration of various sensors. Recognizing that humans perceive the physical environment by acquiring diverse tactile information such as texture and pressure changes, we further propose to learn unified multi-sensor representations from both static and dynamic perspectives. By integrating tactile images and videos, we present AnyTouch, a unified static-dynamic multi-sensor representation learning framework with a multi-level structure, aimed at both enhancing comprehensive perceptual abilities and enabling effective cross-sensor transfer. This multi-level architecture captures pixel-level details from tactile data via masked modeling and enhances perception and transferability by learning semantic-level sensor-agnostic features through multi-modal alignment and cross-sensor matching. We provide a comprehensive analysis of multi-sensor transferability, and validate our method on various datasets and in the real-world pouring task. Experimental results show that our method outperforms existing methods, exhibits outstanding static and dynamic perception capabilities across various sensors. 

**Abstract (ZH)**: 视觉-触觉传感器旨在模拟人类的触觉感知，使机器人能够精确地理解和操作物体。随着时间的推移，众多精心设计的视觉-触觉传感器被集成到机器人系统中，以帮助完成各种任务。然而，这些低标准化的视觉-触觉传感器的特殊数据特征阻碍了强大的触觉感知系统的建立。我们认为解决这一问题的关键在于学习统一的多传感器表示，从而整合传感器并促进它们之间的触觉知识转移。为了实现这种统一表示，我们引入了TacQuad，一个来自四种不同视觉-触觉传感器的对齐多模态多传感器触觉数据集，它允许显式地集成各种传感器。鉴于人类通过获取诸如纹理和压力变化等多种触觉信息来感知物理环境，我们进一步提出从静态和动态两个方面学习统一的多传感器表示。通过整合触觉图像和视频，我们提出了AnyTouch，一个具有多级结构的统一静态-动态多传感器表示学习框架，旨在提高综合感知能力和促进有效的跨传感器转移。该多级架构通过掩蔽建模捕捉触觉数据的像素级细节，并通过多模态对齐和跨传感器匹配学习语义级传感器无损特征，从而增强感知能力和转移性。我们全面分析了多传感器转移能力，并在多种数据集以及现实世界的倒水任务中验证了我们的方法。实验结果表明，我们的方法优于现有方法，在各种传感器上的静态和动态感知能力都表现出色。 

---
# Improved Fine-Tuning of Large Multimodal Models for Hateful Meme Detection 

**Title (ZH)**: 改进的大规模多模态模型细粒度微调在仇恨 meme 检测中的应用 

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2502.13061)  

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While large multimodal models have shown strong generalization across various tasks, they exhibit poor generalization to hateful meme detection due to the dynamic nature of memes tied to emerging social trends and breaking news. Recent work further highlights the limitations of conventional supervised fine-tuning for large multimodal models in this context. To address these challenges, we propose Large Multimodal Model Retrieval-Guided Contrastive Learning (LMM-RGCL), a novel two-stage fine-tuning framework designed to improve both in-domain accuracy and cross-domain generalization. Experimental results on six widely used meme classification datasets demonstrate that LMM-RGCL achieves state-of-the-art performance, outperforming agent-based systems such as VPD-PALI-X-55B. Furthermore, our method effectively generalizes to out-of-domain memes under low-resource settings, surpassing models like GPT-4o. 

**Abstract (ZH)**: 大规模多模态模型检索引导对比学习（LMM-RGCL）：一种改进领域内准确性和跨领域泛化的新型两阶段微调框架 

---
# DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning 

**Title (ZH)**: DeepResonance：通过以音乐为中心的多路指令调优增强多模态音乐理解 

**Authors**: Zhuoyuan Mao, Mengjie Zhao, Qiyu Wu, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2502.12623)  

**Abstract**: Recent advancements in music large language models (LLMs) have significantly improved music understanding tasks, which involve the model's ability to analyze and interpret various musical elements. These improvements primarily focused on integrating both music and text inputs. However, the potential of incorporating additional modalities such as images, videos and textual music features to enhance music understanding remains unexplored. To bridge this gap, we propose DeepResonance, a multimodal music understanding LLM fine-tuned via multi-way instruction tuning with multi-way aligned music, text, image, and video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and Music4way-Any2T, three 4-way training and evaluation datasets designed to enable DeepResonance to integrate both visual and textual music feature content. We also introduce multi-sampled ImageBind embeddings and a pre-alignment Transformer to enhance modality fusion prior to input into text LLMs, tailoring DeepResonance for multi-way instruction tuning. Our model achieves state-of-the-art performances across six music understanding tasks, highlighting the benefits of the auxiliary modalities and the structural superiority of DeepResonance. We plan to open-source the models and the newly constructed datasets. 

**Abstract (ZH)**: Recent advancements in音乐大型语言模型（LLMs）显著提高了音乐理解任务的能力，这些任务涉及模型分析和解释各种音乐元素的能力。这些进步主要集中在整合音乐和文本输入上。然而，将额外的模态，如图像、视频和文本音乐特征纳入以增强音乐理解的可能性仍未被探索。为了解决这一缺口，我们提出了DeepResonance，这是一种通过多维指令调谐和多模态对齐的音乐、文本、图像和视频数据微调的多模态音乐理解LLM。为此，我们构建了Music4way-MI2T、Music4way-MV2T和Music4way-Any2T三个4维训练和评估数据集，旨在使DeepResonance能够整合视觉和文本音乐特征内容。我们还引入了多采样的ImageBind嵌入和预对齐 Transformer，以增强输入到文本LLM之前的模态融合，使DeepResonance适配多维指令调谐。我们的模型在六项音乐理解任务中实现了最先进的性能，突显了辅助模态和DeepResonance结构上的优势。我们计划开源这些模型和新构建的数据集。 

---
# A Comprehensive Survey on Generative AI for Video-to-Music Generation 

**Title (ZH)**: 全面综述：基于视频自动生成音乐的生成型AI技术 

**Authors**: Shulei Ji, Songruoyao Wu, Zihao Wang, Shuyu Li, Kejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12489)  

**Abstract**: The burgeoning growth of video-to-music generation can be attributed to the ascendancy of multimodal generative models. However, there is a lack of literature that comprehensively combs through the work in this field. To fill this gap, this paper presents a comprehensive review of video-to-music generation using deep generative AI techniques, focusing on three key components: visual feature extraction, music generation frameworks, and conditioning mechanisms. We categorize existing approaches based on their designs for each component, clarifying the roles of different strategies. Preceding this, we provide a fine-grained classification of video and music modalities, illustrating how different categories influence the design of components within the generation pipelines. Furthermore, we summarize available multimodal datasets and evaluation metrics while highlighting ongoing challenges in the field. 

**Abstract (ZH)**: 视频到音乐生成的蓬勃发展可归因于多模态生成模型的兴起。然而，缺乏对该领域工作的全面综述文献。为填补这一空白，本文基于深度生成AI技术对视频到音乐生成进行了全面综述，重点讨论三个关键组件：视觉特征提取、音乐生成框架和条件机制。我们根据每个组件的设计将其现有方法分类，澄清不同策略的作用。在此之前，我们提供了视频和音乐模态的精细分类，说明不同类别如何影响生成管道中组件的设计。此外，我们总结了可用的多模态数据集和评估指标，并强调了该领域面临的持续挑战。 

---
# ClusMFL: A Cluster-Enhanced Framework for Modality-Incomplete Multimodal Federated Learning in Brain Imaging Analysis 

**Title (ZH)**: ClusMFL：一种用于脑成像分析的模态不全多模态联邦学习聚类增强框架 

**Authors**: Xinpeng Wang, Rong Zhou, Han Xie, Xiaoying Tang, Lifang He, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12180)  

**Abstract**: Multimodal Federated Learning (MFL) has emerged as a promising approach for collaboratively training multimodal models across distributed clients, particularly in healthcare domains. In the context of brain imaging analysis, modality incompleteness presents a significant challenge, where some institutions may lack specific imaging modalities (e.g., PET, MRI, or CT) due to privacy concerns, device limitations, or data availability issues. While existing work typically assumes modality completeness or oversimplifies missing-modality scenarios, we simulate a more realistic setting by considering both client-level and instance-level modality incompleteness in this study. Building on this realistic simulation, we propose ClusMFL, a novel MFL framework that leverages feature clustering for cross-institutional brain imaging analysis under modality incompleteness. Specifically, ClusMFL utilizes the FINCH algorithm to construct a pool of cluster centers for the feature embeddings of each modality-label pair, effectively capturing fine-grained data distributions. These cluster centers are then used for feature alignment within each modality through supervised contrastive learning, while also acting as proxies for missing modalities, allowing cross-modal knowledge transfer. Furthermore, ClusMFL employs a modality-aware aggregation strategy, further enhancing the model's performance in scenarios with severe modality incompleteness. We evaluate the proposed framework on the ADNI dataset, utilizing structural MRI and PET scans. Extensive experimental results demonstrate that ClusMFL achieves state-of-the-art performance compared to various baseline methods across varying levels of modality incompleteness, providing a scalable solution for cross-institutional brain imaging analysis. 

**Abstract (ZH)**: 多模态联邦学习（MFL）已成为在分布式客户端跨域协作训练多模态模型的有前途的方法，特别是在医疗保健领域。在脑成像分析的背景下，模态不完整提出了一项重大挑战，一些机构可能由于隐私问题、设备限制或数据可用性问题缺少特定的成像模态（如PET、MRI或CT）。虽然现有工作通常假设模态完整性或过度简化缺失模态的情况，本研究通过考虑客户端级别和实例级别模态不完整性来模拟更现实的场景。在此现实模拟的基础上，我们提出了一种新颖的ClusMFL框架，该框架利用特征聚类在模态不完整的情况下进行跨机构脑成像分析。具体而言，ClusMFL利用FINCH算法为每个模态-标签对构建特征嵌入的聚类中心池，有效捕获了细粒度的数据分布。这些聚类中心随后通过监督对比学习在每个模态内进行特征对齐，并作为缺失模态的代理，使跨模态知识转移成为可能。此外，ClusMFL采用模态感知的聚合策略，进一步增强模型在严重模态不完整情况下的性能。我们使用ADNI数据集评估了所提出的框架，该数据集利用了结构MRI和PET扫描。广泛的实验结果表明，ClusMFL在不同程度的模态不完整情况下均取得了最先进的性能，为跨机构脑成像分析提供了可扩展的解决方案。 

---
