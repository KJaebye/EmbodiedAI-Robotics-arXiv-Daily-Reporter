# MambaPlace:Text-to-Point-Cloud Cross-Modal Place Recognition with Attention Mamba Mechanisms 

**Title (ZH)**: MambaPlace：基于注意力Mamba机制的跨模态文本到点云地方识别 

**Authors**: Tianyi Shang, Zhenyu Li, Pengjie Xu, Jinwei Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2408.15740)  

**Abstract**: Vision Language Place Recognition (VLVPR) enhances robot localization performance by incorporating natural language descriptions from images. By utilizing language information, VLVPR directs robot place matching, overcoming the constraint of solely depending on vision. The essence of multimodal fusion lies in mining the complementary information between different modalities. However, general fusion methods rely on traditional neural architectures and are not well equipped to capture the dynamics of cross modal interactions, especially in the presence of complex intra modal and inter modal correlations. To this end, this paper proposes a novel coarse to fine and end to end connected cross modal place recognition framework, called MambaPlace. In the coarse localization stage, the text description and 3D point cloud are encoded by the pretrained T5 and instance encoder, respectively. They are then processed using Text Attention Mamba (TAM) and Point Clouds Mamba (PCM) for data enhancement and alignment. In the subsequent fine localization stage, the features of the text description and 3D point cloud are cross modally fused and further enhanced through cascaded Cross Attention Mamba (CCAM). Finally, we predict the positional offset from the fused text point cloud features, achieving the most accurate localization. Extensive experiments show that MambaPlace achieves improved localization accuracy on the KITTI360Pose dataset compared to the state of the art methods. 

**Abstract (ZH)**: Vision-Language-Powered Place Recognition (VLVPR) 提高了机器人定位性能通过结合图像中的自然语言描述。通过利用语言信息，VLVPR 引导机器人位置匹配，克服了仅仅依赖视觉的限制。多模态融合的本质在于挖掘不同模态之间的互补信息。然而，一般的融合方法依赖于传统的神经架构，并不擅长捕捉跨模态交互的动态特性，尤其是在存在复杂内模态和跨模态相关性的情况下。为此，本文提出了一种从粗到细且端到端连接的跨模态位置识别框架，称为 MambaPlace。在粗定位阶段，文本描述和 3D 点云分别由预训练的 T5 和实例编码器编码。然后使用文本注意力 Mamba (TAM) 和点云 Mamba (PCM) 对数据进行增强和对齐。在后续的精细定位阶段，文本描述和 3D 点云的特征通过级联跨注意力 Mamba (CCAM) 跨模态融合并进一步增强。最后，我们从融合的文本点云特征中预测位置偏移，实现最准确的定位。大量实验表明，MambaPlace 在 KITTI360Pose 数据集上实现了比现有方法更高的定位精度。 

---
# Bridging Domain Gaps between Pretrained Multimodal Models and Recommendations 

**Title (ZH)**: 预训练多模态模型与推荐系统之间的域差距桥梁构建 

**Authors**: Wenyu Zhang, Jie Luo, Xinming Zhang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15542)  

**Abstract**: With the explosive growth of multimodal content online, pre-trained visual-language models have shown great potential for multimodal recommendation. However, while these models achieve decent performance when applied in a frozen manner, surprisingly, due to significant domain gaps (e.g., feature distribution discrepancy and task objective misalignment) between pre-training and personalized recommendation, adopting a joint training approach instead leads to performance worse than baseline. Existing approaches either rely on simple feature extraction or require computationally expensive full model fine-tuning, struggling to balance effectiveness and efficiency. To tackle these challenges, we propose \textbf{P}arameter-efficient \textbf{T}uning for \textbf{M}ultimodal \textbf{Rec}ommendation (\textbf{PTMRec}), a novel framework that bridges the domain gap between pre-trained models and recommendation systems through a knowledge-guided dual-stage parameter-efficient training strategy. This framework not only eliminates the need for costly additional pre-training but also flexibly accommodates various parameter-efficient tuning methods. 

**Abstract (ZH)**: 参数高效调 tune 多模态推荐（PTMRec） 

---
# MVIP -- A Dataset and Methods for Application Oriented Multi-View and Multi-Modal Industrial Part Recognition 

**Title (ZH)**: MVIP -- 一种面向应用的多视图多模态工业零件识别数据集和方法 

**Authors**: Paul Koch, Marian Schlüter, Jörg Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2502.15448)  

**Abstract**: We present MVIP, a novel dataset for multi-modal and multi-view application-oriented industrial part recognition. Here we are the first to combine a calibrated RGBD multi-view dataset with additional object context such as physical properties, natural language, and super-classes. The current portfolio of available datasets offers a wide range of representations to design and benchmark related methods. In contrast to existing classification challenges, industrial recognition applications offer controlled multi-modal environments but at the same time have different problems than traditional 2D/3D classification challenges. Frequently, industrial applications must deal with a small amount or increased number of training data, visually similar parts, and varying object sizes, while requiring a robust near 100% top 5 accuracy under cost and time constraints. Current methods tackle such challenges individually, but direct adoption of these methods within industrial applications is complex and requires further research. Our main goal with MVIP is to study and push transferability of various state-of-the-art methods within related downstream tasks towards an efficient deployment of industrial classifiers. Additionally, we intend to push with MVIP research regarding several modality fusion topics, (automated) synthetic data generation, and complex data sampling -- combined in a single application-oriented benchmark. 

**Abstract (ZH)**: MVIP：面向多模态多视角工业部件识别的新型数据集 

---
# Evaluating Multimodal Generative AI with Korean Educational Standards 

**Title (ZH)**: 基于韩国教育标准评估多模态生成人工智能 

**Authors**: Sanghee Park, Geewook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.15422)  

**Abstract**: This paper presents the Korean National Educational Test Benchmark (KoNET), a new benchmark designed to evaluate Multimodal Generative AI Systems using Korean national educational tests. KoNET comprises four exams: the Korean Elementary General Educational Development Test (KoEGED), Middle (KoMGED), High (KoHGED), and College Scholastic Ability Test (KoCSAT). These exams are renowned for their rigorous standards and diverse questions, facilitating a comprehensive analysis of AI performance across different educational levels. By focusing on Korean, KoNET provides insights into model performance in less-explored languages. We assess a range of models - open-source, open-access, and closed APIs - by examining difficulties, subject diversity, and human error rates. The code and dataset builder will be made fully open-sourced at this https URL. 

**Abstract (ZH)**: 韩国国家教育测试基准（KoNET）：面向韩语文本的多模态生成人工智能系统评估基准 

---
# Beyond Words: Exploring Cultural Value Sensitivity in Multimodal Models 

**Title (ZH)**: 超越文字：探索多模态模型中的文化价值敏感性 

**Authors**: Srishti Yadav, Zhi Zhang, Daniel Hershcovich, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2502.14906)  

**Abstract**: Investigating value alignment in Large Language Models (LLMs) based on cultural context has become a critical area of research. However, similar biases have not been extensively explored in large vision-language models (VLMs). As the scale of multimodal models continues to grow, it becomes increasingly important to assess whether images can serve as reliable proxies for culture and how these values are embedded through the integration of both visual and textual data. In this paper, we conduct a thorough evaluation of multimodal model at different scales, focusing on their alignment with cultural values. Our findings reveal that, much like LLMs, VLMs exhibit sensitivity to cultural values, but their performance in aligning with these values is highly context-dependent. While VLMs show potential in improving value understanding through the use of images, this alignment varies significantly across contexts highlighting the complexities and underexplored challenges in the alignment of multimodal models. 

**Abstract (ZH)**: 基于文化背景探究大型语言模型的价值对齐已成为一个重要研究领域。然而，类似偏见在大型视觉-语言模型(VLMs)中的研究尚不充分。随着多模态模型规模的不断扩大，评估图像是否能可靠地代表文化以及这些价值是如何通过视觉和文本数据的结合而嵌入变得尤为重要。在本文中，我们针对不同规模的多模态模型进行了详细的评估，重点关注它们与文化价值的对齐情况。我们的研究发现，与大型语言模型(LLMs)类似，视觉-语言模型也表现出对文化价值的敏感性，但它们在这些价值上的表现高度依赖于上下文。尽管视觉-语言模型通过使用图像提高价值理解具有潜力，但这种对齐在不同上下文中的差异性揭示了多模态模型对齐中的复杂性和未被充分探索的挑战。 

---
# NOTA: Multimodal Music Notation Understanding for Visual Large Language Model 

**Title (ZH)**: 多模态音乐符号理解对于视觉大语言模型 

**Authors**: Mingni Tang, Jiajia Li, Lu Yang, Zhiqiang Zhang, Jinghao Tian, Zuchao Li, Lefei Zhang, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14893)  

**Abstract**: Symbolic music is represented in two distinct forms: two-dimensional, visually intuitive score images, and one-dimensional, standardized text annotation sequences. While large language models have shown extraordinary potential in music, current research has primarily focused on unimodal symbol sequence text. Existing general-domain visual language models still lack the ability of music notation understanding. Recognizing this gap, we propose NOTA, the first large-scale comprehensive multimodal music notation dataset. It consists of 1,019,237 records, from 3 regions of the world, and contains 3 tasks. Based on the dataset, we trained NotaGPT, a music notation visual large language model. Specifically, we involve a pre-alignment training phase for cross-modal alignment between the musical notes depicted in music score images and their textual representation in ABC notation. Subsequent training phases focus on foundational music information extraction, followed by training on music notation analysis. Experimental results demonstrate that our NotaGPT-7B achieves significant improvement on music understanding, showcasing the effectiveness of NOTA and the training pipeline. Our datasets are open-sourced at this https URL. 

**Abstract (ZH)**: 符号音乐以两种不同的形式表示：二维、直观的乐谱图像和一维、标准化的文字注释序列。虽然大型语言模型在音乐领域展现了非凡的潜力，但现有研究主要集中在单一符号序列文本上。现有的通用视觉语言模型在音乐符号理解方面仍缺乏能力。认识到这一缺口，我们提出了NOTA，这是首个大规模综合多模态音乐符号数据集，包含1,019,237条记录，来自世界三个地区，并包含3项任务。基于此数据集，我们训练了Notagpt，这是一种音乐符号视觉大型语言模型。特别地，我们通过一个预对齐训练阶段实现音乐得分图像中表示的音符与其ABC符号表示之间的跨模态对齐。后续训练阶段专注于基础音乐信息提取，随后进行音乐符号分析。实验结果表明，我们的Notagpt-7B在音乐理解方面取得了显著改进，展示了NOTA和训练管道的有效性。我们的数据集在此处开源<https://>。 

---
# EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild 

**Title (ZH)**: EgoSpeak: 学习自中心对话代理在真实世界中何时发言 

**Authors**: Junhyeok Kim, Min Soo Kim, Jiwan Chung, Jungbin Cho, Jisoo Kim, Sungwoong Kim, Gyeongbo Sim, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14892)  

**Abstract**: Predicting when to initiate speech in real-world environments remains a fundamental challenge for conversational agents. We introduce EgoSpeak, a novel framework for real-time speech initiation prediction in egocentric streaming video. By modeling the conversation from the speaker's first-person viewpoint, EgoSpeak is tailored for human-like interactions in which a conversational agent must continuously observe its environment and dynamically decide when to talk. Our approach bridges the gap between simplified experimental setups and complex natural conversations by integrating four key capabilities: (1) first-person perspective, (2) RGB processing, (3) online processing, and (4) untrimmed video processing. We also present YT-Conversation, a diverse collection of in-the-wild conversational videos from YouTube, as a resource for large-scale pretraining. Experiments on EasyCom and Ego4D demonstrate that EgoSpeak outperforms random and silence-based baselines in real time. Our results also highlight the importance of multimodal input and context length in effectively deciding when to speak. 

**Abstract (ZH)**: 在真实环境中的语音发起预测仍然是对话代理的基本挑战。我们介绍了一种新颖的框架EgoSpeak，用于自视点流式视频中的实时语音发起预测。通过从说话人的第一人称视角建模对话，EgoSpeak专为人类般的交互设计，其中对话代理必须不断观察环境并在适当时机决定是否发言。我们的方法通过结合四种关键能力，弥合了简化实验设置与复杂自然对话之间的差距：（1）第一人称视角，（2）RGB处理，（3）在线处理，以及（4）未剪辑视频处理。我们还介绍了YT-Conversation，这是一个来自YouTube的多元化的在野对话视频集合，作为大规模预训练的资源。在EasyCom和Ego4D上的实验表明，EgoSpeak在实时性能上优于随机和静默基线。我们的结果还强调了多模态输入和上下文长度在有效决定何时发言中的重要性。 

---
# Narrowing Information Bottleneck Theory for Multimodal Image-Text Representations Interpretability 

**Title (ZH)**: 窄信息瓶颈理论在多模态图像-文本表征可解释性中的应用 

**Authors**: Zhiyu Zhu, Zhibo Jin, Jiayu Zhang, Nan Yang, Jiahao Huang, Jianlong Zhou, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.14889)  

**Abstract**: The task of identifying multimodal image-text representations has garnered increasing attention, particularly with models such as CLIP (Contrastive Language-Image Pretraining), which demonstrate exceptional performance in learning complex associations between images and text. Despite these advancements, ensuring the interpretability of such models is paramount for their safe deployment in real-world applications, such as healthcare. While numerous interpretability methods have been developed for unimodal tasks, these approaches often fail to transfer effectively to multimodal contexts due to inherent differences in the representation structures. Bottleneck methods, well-established in information theory, have been applied to enhance CLIP's interpretability. However, they are often hindered by strong assumptions or intrinsic randomness. To overcome these challenges, we propose the Narrowing Information Bottleneck Theory, a novel framework that fundamentally redefines the traditional bottleneck approach. This theory is specifically designed to satisfy contemporary attribution axioms, providing a more robust and reliable solution for improving the interpretability of multimodal models. In our experiments, compared to state-of-the-art methods, our approach enhances image interpretability by an average of 9%, text interpretability by an average of 58.83%, and accelerates processing speed by 63.95%. Our code is publicly accessible at this https URL. 

**Abstract (ZH)**: 多模态图像-文本表示的识别任务引起了越来越多的关注，特别是CLIP（对比语言图像预训练）等模型在学习图像和文本之间复杂关联方面表现出色。尽管取得了这些进步，确保这些模型的安全可解释性对于其在医疗保健等实际应用中的部署至关重要。尽管已经为单模态任务开发了多种可解释性方法，但由于表示结构的固有差异，这些方法往往难以有效转移到多模态上下文。信息理论中成熟的瓶颈方法已被应用于提高CLIP的可解释性，但这些方法经常受到强烈假设或内在随机性的阻碍。为克服这些挑战，我们提出了信息瓶颈压缩理论这一新框架，从根本上重新定义了传统瓶颈方法。该理论专门为符合当前归因公理设计，为提高多模态模型的可解释性提供了更稳健且可靠的方法。在我们的实验中，与最先进的方法相比，我们的方法平均提高了图象可解释性9%，文本可解释性58.83%，并加速了处理速度63.95%。我们的代码可在以下网址公开访问：这个 https URL。 

---
# The Multi-Faceted Monosemanticity in Multimodal Representations 

**Title (ZH)**: 多模态表示中的多面向单义性 

**Authors**: Hanqi Yan, Xiangxiang Cui, Lu Yin, Paul Pu Liang, Yulan He, Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14888)  

**Abstract**: In this paper, we leverage recent advancements in feature monosemanticity to extract interpretable features from deep multimodal models, offering a data-driven understanding of modality gaps. Specifically, we investigate CLIP (Contrastive Language-Image Pretraining), a prominent visual-language representation model trained on extensive image-text pairs. Building upon interpretability tools developed for single-modal models, we extend these methodologies to assess multi-modal interpretability of CLIP features. Additionally, we introduce the Modality Dominance Score (MDS) to attribute the interpretability of each feature to its respective modality. Next, we transform CLIP features into a more interpretable space, enabling us to categorize them into three distinct classes: vision features (single-modal), language features (single-modal), and visual-language features (cross-modal). Our findings reveal that this categorization aligns closely with human cognitive understandings of different modalities. We also demonstrate significant use cases of this modality-specific features including detecting gender bias, adversarial attack defense and text-to-image model editing. These results indicate that large-scale multimodal models, equipped with task-agnostic interpretability tools, offer valuable insights into key connections and distinctions between different modalities. 

**Abstract (ZH)**: 利用特征单义性从深度多模态模型中提取可解释特征，探究模态差距的数据驱动理解：CLIP的多模态可解释性分析与模态主导得分应用 

---
