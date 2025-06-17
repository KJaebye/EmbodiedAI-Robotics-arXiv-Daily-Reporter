# CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding 

**Title (ZH)**: CEED-VLA：具早期退出解码的一致性跨模态模型 

**Authors**: Wenxuan Song, Jiayi Chen, Pengxiang Ding, Yuxin Huang, Han Zhao, Donglin Wang, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13725)  

**Abstract**: In recent years, Vision-Language-Action (VLA) models have become a vital research direction in robotics due to their impressive multimodal understanding and generalization capabilities. Despite the progress, their practical deployment is severely constrained by inference speed bottlenecks, particularly in high-frequency and dexterous manipulation tasks. While recent studies have explored Jacobi decoding as a more efficient alternative to traditional autoregressive decoding, its practical benefits are marginal due to the lengthy iterations. To address it, we introduce consistency distillation training to predict multiple correct action tokens in each iteration, thereby achieving acceleration. Besides, we design mixed-label supervision to mitigate the error accumulation during distillation. Although distillation brings acceptable speedup, we identify that certain inefficient iterations remain a critical bottleneck. To tackle this, we propose an early-exit decoding strategy that moderately relaxes convergence conditions, which further improves average inference efficiency. Experimental results show that the proposed method achieves more than 4 times inference acceleration across different baselines while maintaining high task success rates in both simulated and real-world robot tasks. These experiments validate that our approach provides an efficient and general paradigm for accelerating multimodal decision-making in robotics. Our project page is available at this https URL. 

**Abstract (ZH)**: 近年来，视觉-语言-动作（VLA）模型由于其令人印象深刻的跨模态理解和泛化能力，已经成为机器人领域的一个重要研究方向。尽管取得了进展，但在高频灵巧操作任务中，其实际部署仍然受到推理速度瓶颈的严重制约。虽然近期研究探索了雅可比解码作为传统自回归解码的更高效替代方法，但由于迭代过程较长，其实际优势有限。为了解决这一问题，我们引入一致性蒸馏训练，在每次迭代中预测多个正确的动作令牌，从而实现加速。此外，我们设计了混合标签监督，以减轻蒸馏过程中的错误累积。尽管蒸馏带来了可接受的加速，但我们发现某些不高效的迭代仍然是关键瓶颈。为解决这一问题，我们提出了一种早期退出解码策略，适度放宽收敛条件，从而进一步提高平均推理效率。实验结果表明，所提出的方法在不同baseline上实现了超过4倍的推理加速，同时在模拟和真实世界机器人任务中保持了较高的任务成功率。这些实验验证了我们的方法为机器人领域加速跨模态决策提供了一种高效且通用的范式。我们的项目页面可以在以下链接访问：this https URL。 

---
# Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model 

**Title (ZH)**: Stream-Omi：大型语言-视觉-语音模型下的多模态同时交互 

**Authors**: Shaolei Zhang, Shoutao Guo, Qingkai Fang, Yan Zhou, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13642)  

**Abstract**: The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience. 

**Abstract (ZH)**: GPT-4o-like大型多模态模型的 emergence 及其对文本、视觉和语音模态整合的探索：一种高效模态对齐的 Stream-Omni 模型 

---
# Rethinking Explainability in the Era of Multimodal AI 

**Title (ZH)**: 重新思考多模态AI时代的可解释性 

**Authors**: Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.13060)  

**Abstract**: While multimodal AI systems (models jointly trained on heterogeneous data types such as text, time series, graphs, and images) have become ubiquitous and achieved remarkable performance across high-stakes applications, transparent and accurate explanation algorithms are crucial for their safe deployment and ensure user trust. However, most existing explainability techniques remain unimodal, generating modality-specific feature attributions, concepts, or circuit traces in isolation and thus failing to capture cross-modal interactions. This paper argues that such unimodal explanations systematically misrepresent and fail to capture the cross-modal influence that drives multimodal model decisions, and the community should stop relying on them for interpreting multimodal models. To support our position, we outline key principles for multimodal explanations grounded in modality: Granger-style modality influence (controlled ablations to quantify how removing one modality changes the explanation for another), Synergistic faithfulness (explanations capture the model's predictive power when modalities are combined), and Unified stability (explanations remain consistent under small, cross-modal perturbations). This targeted shift to multimodal explanations will help the community uncover hidden shortcuts, mitigate modality bias, improve model reliability, and enhance safety in high-stakes settings where incomplete explanations can have serious consequences. 

**Abstract (ZH)**: 多模态AI系统透明性和准确解释算法：从单模态到多模态解释的转变 

---
# MM-R5: MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval 

**Title (ZH)**: MM-R5: 基于强化学习的多模态推理排序器用于文档检索 

**Authors**: Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin Zhong, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.12364)  

**Abstract**: Multimodal document retrieval systems enable information access across text, images, and layouts, benefiting various domains like document-based question answering, report analysis, and interactive content summarization. Rerankers improve retrieval precision by reordering retrieved candidates. However, current multimodal reranking methods remain underexplored, with significant room for improvement in both training strategies and overall effectiveness. Moreover, the lack of explicit reasoning makes it difficult to analyze and optimize these methods further. In this paper, We propose MM-R5, a MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval, aiming to provide a more effective and reliable solution for multimodal reranking tasks. MM-R5 is trained in two stages: supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we focus on improving instruction-following and guiding the model to generate complete and high-quality reasoning chains. To support this, we introduce a novel data construction strategy that produces rich, high-quality reasoning data. In the RL stage, we design a task-specific reward framework, including a reranking reward tailored for multimodal candidates and a composite template-based reward to further refine reasoning quality. We conduct extensive experiments on MMDocIR, a challenging public benchmark spanning multiple domains. MM-R5 achieves state-of-the-art performance on most metrics and delivers comparable results to much larger models on the remaining ones. Moreover, compared to the best retrieval-only method, MM-R5 improves recall@1 by over 4%. These results validate the effectiveness of our reasoning-enhanced training pipeline. 

**Abstract (ZH)**: 多模态文档检索系统 enables 信息访问跨越文本、图像和布局，惠及文档为基础的问题回答、报告分析和互动内容总结等多个领域。排序模型通过重新排序检索候选项以提高检索精度。然而，当前的多模态排序模型仍存在较大探索空间，特别是在训练策略和整体效果方面改进空间巨大。此外，缺乏明确的推理过程使得这些方法的进一步分析和优化变得困难。本文提出 MM-R5，一种基于强化学习的多模态增强排序器，旨在为多模态排序任务提供更为有效和可靠的方法。MM-R5 在两个阶段进行训练：监督微调 (SFT) 和强化学习 (RL)。在 SFT 阶段，我们专注于提高指令遵循能力，并引导模型生成完整且高质量的推理链。为此，我们引入了一种新颖的数据构造策略，以生成丰富且高质量的推理数据。在 RL 阶段，我们设计了一种特定任务的奖励框架，包括针对多模态候选项的排序奖励和基于复合模板的奖励，以进一步提高推理质量。我们针对 MMDocIR 这一具有挑战性的跨领域公开基准进行了广泛实验。MM-R5 在大部分指标上取得了最先进的性能，在部分指标上与更大规模的模型实现可比的结果。此外，相比仅基于检索的最佳方法，MM-R5 将召回率@1 提高了超过 4%。这些结果验证了我们增强推理训练框架的有效性。 

---
# Active Multimodal Distillation for Few-shot Action Recognition 

**Title (ZH)**: 面向Few-shot动作识别的主动多模态蒸馏 

**Authors**: Weijia Feng, Yichen Zhu, Ruojia Zhang, Chenyang Wang, Fei Ma, Xiaobao Wang, Xiaobai Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13322)  

**Abstract**: Owing to its rapid progress and broad application prospects, few-shot action recognition has attracted considerable interest. However, current methods are predominantly based on limited single-modal data, which does not fully exploit the potential of multimodal information. This paper presents a novel framework that actively identifies reliable modalities for each sample using task-specific contextual cues, thus significantly improving recognition performance. Our framework integrates an Active Sample Inference (ASI) module, which utilizes active inference to predict reliable modalities based on posterior distributions and subsequently organizes them accordingly. Unlike reinforcement learning, active inference replaces rewards with evidence-based preferences, making more stable predictions. Additionally, we introduce an active mutual distillation module that enhances the representation learning of less reliable modalities by transferring knowledge from more reliable ones. Adaptive multimodal inference is employed during the meta-test to assign higher weights to reliable modalities. Extensive experiments across multiple benchmarks demonstrate that our method significantly outperforms existing approaches. 

**Abstract (ZH)**: 由于其快速进步和广泛的应用前景，少样本动作识别引起了 considerable attention。然而，当前方法主要基于有限的单模数据，未能充分挖掘多模信息的潜力。本文提出了一种新颖的框架，该框架能够利用任务特定的上下文线索主动识别每个样本的可靠模态，从而显著提高识别性能。我们的框架整合了一个主动样本推理（ASI）模块，该模块利用主动推理根据后验分布预测可靠模态并相应地进行组织。与强化学习不同，主动推理使用证据为基础的偏好替代奖励，从而做出更为稳定的预测。此外，我们引入了一个主动互信息蒸馏模块，通过从更可靠模态转移知识来增强不可靠模态的表示学习。在元测试过程中采用自适应多模态推理，在分配权重时给予可靠模态更高的权重。在多个基准上的广泛实验表明，我们的方法显著优于现有方法。 

---
# NAP-Tuning: Neural Augmented Prompt Tuning for Adversarially Robust Vision-Language Models 

**Title (ZH)**: NAP调优：神经增强提示调优以提高对抗 robust 的视觉语言模型性能 

**Authors**: Jiaming Zhang, Xin Wang, Xingjun Ma, Lingyu Qiu, Yu-Gang Jiang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12706)  

**Abstract**: Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capabilities in understanding relationships between visual and textual data through joint embedding spaces. Despite their effectiveness, these models remain vulnerable to adversarial attacks, particularly in the image modality, posing significant security concerns. Building upon our previous work on Adversarial Prompt Tuning (AdvPT), which introduced learnable text prompts to enhance adversarial robustness in VLMs without extensive parameter training, we present a significant extension by introducing the Neural Augmentor framework for Multi-modal Adversarial Prompt Tuning (NAP-Tuning).Our key innovations include: (1) extending AdvPT from text-only to multi-modal prompting across both text and visual modalities, (2) expanding from single-layer to multi-layer prompt architectures, and (3) proposing a novel architecture-level redesign through our Neural Augmentor approach, which implements feature purification to directly address the distortions introduced by adversarial attacks in feature space. Our NAP-Tuning approach incorporates token refiners that learn to reconstruct purified features through residual connections, allowing for modality-specific and layer-specific feature this http URL experiments demonstrate that NAP-Tuning significantly outperforms existing methods across various datasets and attack types. Notably, our approach shows significant improvements over the strongest baselines under the challenging AutoAttack benchmark, outperforming them by 33.5% on ViT-B16 and 33.0% on ViT-B32 architectures while maintaining competitive clean accuracy. 

**Abstract (ZH)**: Vision-语言模型多模态对抗提示调优框架（NAP-Tuning） 

---
