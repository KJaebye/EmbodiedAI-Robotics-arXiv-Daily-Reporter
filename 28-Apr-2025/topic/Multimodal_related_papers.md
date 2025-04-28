# A Multimodal Deep Learning Approach for White Matter Shape Prediction in Diffusion MRI Tractography 

**Title (ZH)**: 多模态深度学习方法在扩散磁共振成像tractography中白质形状预测 

**Authors**: Yui Lo, Yuqian Chen, Dongnan Liu, Leo Zekelman, Jarrett Rushmore, Yogesh Rathi, Nikos Makris, Alexandra J. Golby, Fan Zhang, Weidong Cai, Lauren J. O'Donnell  

**Link**: [PDF](https://arxiv.org/pdf/2504.18400)  

**Abstract**: Shape measures have emerged as promising descriptors of white matter tractography, offering complementary insights into anatomical variability and associations with cognitive and clinical phenotypes. However, conventional methods for computing shape measures are computationally expensive and time-consuming for large-scale datasets due to reliance on voxel-based representations. We propose Tract2Shape, a novel multimodal deep learning framework that leverages geometric (point cloud) and scalar (tabular) features to predict ten white matter tractography shape measures. To enhance model efficiency, we utilize a dimensionality reduction algorithm for the model to predict five primary shape components. The model is trained and evaluated on two independently acquired datasets, the HCP-YA dataset, and the PPMI dataset. We evaluate the performance of Tract2Shape by training and testing it on the HCP-YA dataset and comparing the results with state-of-the-art models. To further assess its robustness and generalization ability, we also test Tract2Shape on the unseen PPMI dataset. Tract2Shape outperforms SOTA deep learning models across all ten shape measures, achieving the highest average Pearson's r and the lowest nMSE on the HCP-YA dataset. The ablation study shows that both multimodal input and PCA contribute to performance gains. On the unseen testing PPMI dataset, Tract2Shape maintains a high Pearson's r and low nMSE, demonstrating strong generalizability in cross-dataset evaluation. Tract2Shape enables fast, accurate, and generalizable prediction of white matter shape measures from tractography data, supporting scalable analysis across datasets. This framework lays a promising foundation for future large-scale white matter shape analysis. 

**Abstract (ZH)**: 形状度量已作为白质追踪的有希望的描述符出现，提供了关于解剖变异性和与认知和临床表型关联的补充洞见。然而，传统形状度量计算方法因依赖体素表示而在大规模数据集中计算昂贵且耗时。我们提出了一种名为Tract2Shape的新型多模态深度学习框架，利用几何（点云）和标量（表格）特征来预测十种白质追踪的形状度量。为了提高模型效率，我们利用降维算法，使模型能够预测五个主要形状组件。该模型在HCP-YA数据集和PPMI数据集上进行了训练和评估。我们通过在HCP-YA数据集上训练和测试Tract2Shape，并与最先进的模型进行比较，来评估其性能。为进一步评估其鲁棒性和泛化能力，我们也在未见的PPMI数据集上测试了Tract2Shape。Tract2Shape在所有十种形状度量上均优于最新的深度学习模型，在HCP-YA数据集上实现了最高的平均皮尔森相关系数和最低的nMSE。消融研究显示，多模态输入和主成分分析（PCA）均有助于性能提升。在未见的测试PPMI数据集上，Tract2Shape保持了高皮尔森相关系数和低nMSE，显示出强大的跨数据集泛化能力。Tract2Shape能够快速、准确、泛化地从追踪数据预测白质形状度量，支持跨数据集的可扩展分析。该框架为未来的大型白质形状分析奠定了有希望的基础。 

---
# Seeing Soundscapes: Audio-Visual Generation and Separation from Soundscapes Using Audio-Visual Separator 

**Title (ZH)**: 视听共生：基于视听分离器的声音景观的音视频生成与分离 

**Authors**: Minjae Kang, Martim Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2504.18283)  

**Abstract**: Recent audio-visual generative models have made substantial progress in generating images from audio. However, existing approaches focus on generating images from single-class audio and fail to generate images from mixed audio. To address this, we propose an Audio-Visual Generation and Separation model (AV-GAS) for generating images from soundscapes (mixed audio containing multiple classes). Our contribution is threefold: First, we propose a new challenge in the audio-visual generation task, which is to generate an image given a multi-class audio input, and we propose a method that solves this task using an audio-visual separator. Second, we introduce a new audio-visual separation task, which involves generating separate images for each class present in a mixed audio input. Lastly, we propose new evaluation metrics for the audio-visual generation task: Class Representation Score (CRS) and a modified R@K. Our model is trained and evaluated on the VGGSound dataset. We show that our method outperforms the state-of-the-art, achieving 7% higher CRS and 4% higher R@2* in generating plausible images with mixed audio. 

**Abstract (ZH)**: Recent音频-视觉生成模型已在从音频生成图像方面取得了显著进展。然而，现有方法侧重于从单一类别的音频生成图像，而无法生成从混合音频生成的图像。为解决这一问题，我们提出了一个音频-视觉生成与分离模型（AV-GAS），用于从音景（包含多个类别的混合音频）生成图像。我们的贡献包括三个方面：首先，我们提出了音频-视觉生成任务中的一个新挑战，即给定一个多类别的音频输入生成图像，并提出了一种使用音频-视觉分离器解决问题的方法。其次，我们引入了一个新的音频-视觉分离任务，涉及为混合音频输入中存在的每个类别生成单独的图像。最后，我们提出了音频-视觉生成任务的新评估指标：类表示得分（CRS）和修改后的R@K。我们的模型在VGGSound数据集上进行了训练和评估。结果显示，我们的方法优于现有最佳方法，在从混合音频生成可信图像方面，CRS提高了7%，R@2*提高了4%。 

---
# DMS-Net:Dual-Modal Multi-Scale Siamese Network for Binocular Fundus Image Classification 

**Title (ZH)**: DMS-Net：双模态多尺度孪生网络用于双眼底图像分类 

**Authors**: Guohao Huo, Zibo Lin, Zitong Wang, Ruiting Dai, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18046)  

**Abstract**: Ophthalmic diseases pose a significant global health challenge, yet traditional diagnosis methods and existing single-eye deep learning approaches often fail to account for binocular pathological correlations. To address this, we propose DMS-Net, a dual-modal multi-scale Siamese network for binocular fundus image classification. Our framework leverages weight-shared Siamese ResNet-152 backbones to extract deep semantic features from paired fundus images. To tackle challenges such as lesion boundary ambiguity and scattered pathological distributions, we introduce a Multi-Scale Context-Aware Module (MSCAM) that integrates adaptive pooling and attention mechanisms for multi-resolution feature aggregation. Additionally, a Dual-Modal Feature Fusion (DMFF) module enhances cross-modal interaction through spatial-semantic recalibration and bidirectional attention, effectively combining global context and local edge features. Evaluated on the ODIR-5K dataset, DMS-Net achieves state-of-the-art performance with 80.5% accuracy, 86.1% recall, and 83.8% Cohen's kappa, demonstrating superior capability in detecting symmetric pathologies and advancing clinical decision-making for ocular diseases. 

**Abstract (ZH)**: 双眼视网膜图像分类的双模态多尺度Siamese网络:DMS-Net 

---
# Memory Reviving, Continuing Learning and Beyond: Evaluation of Pre-trained Encoders and Decoders for Multimodal Machine Translation 

**Title (ZH)**: 记忆重现、持续学习及更进一步：预训练编码器和解码器在多模态机器翻译中的评估 

**Authors**: Zhuang Yu, Shiliang Sun, Jing Zhao, Tengfei Song, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18012)  

**Abstract**: Multimodal Machine Translation (MMT) aims to improve translation quality by leveraging auxiliary modalities such as images alongside textual input. While recent advances in large-scale pre-trained language and vision models have significantly benefited unimodal natural language processing tasks, their effectiveness and role in MMT remain underexplored. In this work, we conduct a systematic study on the impact of pre-trained encoders and decoders in multimodal translation models. Specifically, we analyze how different training strategies, from training from scratch to using pre-trained and partially frozen components, affect translation performance under a unified MMT framework. Experiments are carried out on the Multi30K and CoMMuTE dataset across English-German and English-French translation tasks. Our results reveal that pre-training plays a crucial yet asymmetrical role in multimodal settings: pre-trained decoders consistently yield more fluent and accurate outputs, while pre-trained encoders show varied effects depending on the quality of visual-text alignment. Furthermore, we provide insights into the interplay between modality fusion and pre-trained components, offering guidance for future architecture design in multimodal translation systems. 

**Abstract (ZH)**: 多模态机器翻译（MMT）旨在通过利用图像等辅助模态来提高翻译质量。尽管大规模预训练语言和视觉模型在单模态自然语言处理任务中取得了显著进展，但它们在多模态翻译中的有效性及其作用仍然有待探索。在本工作中，我们系统研究了预训练编码器和解码器在多模态翻译模型中的影响。具体而言，我们分析了从从头训练到使用预训练和部分冻结组件的不同训练策略对统一多模态翻译框架下的翻译性能的影响。我们在Multi30K和CoMMuTE数据集上的英语-德语和英语-法语翻译任务中进行了实验。我们的结果表明，预训练在多模态设置中扮演着重要但不对称的角色：预训练解码器始终产生更加流畅和准确的输出，而预训练编码器的效果则取决于视觉-文本对齐的质量。此外，我们探讨了模态融合与预训练组件之间的相互作用，为未来的多模态翻译系统架构设计提供指导。 

---
# FashionM3: Multimodal, Multitask, and Multiround Fashion Assistant based on Unified Vision-Language Model 

**Title (ZH)**: FashionM3：基于统一视觉-语言模型的多模态、多任务、多轮次时尚助手 

**Authors**: Kaicheng Pang, Xingxing Zou, Waikeung Wong  

**Link**: [PDF](https://arxiv.org/pdf/2504.17826)  

**Abstract**: Fashion styling and personalized recommendations are pivotal in modern retail, contributing substantial economic value in the fashion industry. With the advent of vision-language models (VLM), new opportunities have emerged to enhance retailing through natural language and visual interactions. This work proposes FashionM3, a multimodal, multitask, and multiround fashion assistant, built upon a VLM fine-tuned for fashion-specific tasks. It helps users discover satisfying outfits by offering multiple capabilities including personalized recommendation, alternative suggestion, product image generation, and virtual try-on simulation. Fine-tuned on the novel FashionRec dataset, comprising 331,124 multimodal dialogue samples across basic, personalized, and alternative recommendation tasks, FashionM3 delivers contextually personalized suggestions with iterative refinement through multiround interactions. Quantitative and qualitative evaluations, alongside user studies, demonstrate FashionM3's superior performance in recommendation effectiveness and practical value as a fashion assistant. 

**Abstract (ZH)**: 时尚搭配和个人化推荐在现代零售中至关重要，为时尚行业带来了巨大的经济价值。随着视觉语言模型（VLM）的发展，通过自然语言和视觉交互为零售业带来了新的机遇。本文提出了一种基于专门任务微调的视觉语言模型构建的多模态、多任务、多轮次时尚助手——FashionM3。它通过提供个性化推荐、替代建议、产品图像生成和虚拟试穿模拟等多种功能，帮助用户发现满意搭配。FashionM3基于包含331,124个多模态对话样本的新颖FashionRec数据集，涵盖基本推荐、个性化推荐和替代推荐任务，通过多轮次交互提供上下文相关的个性化建议。定量和定性评估，以及用户研究，证明了FashionM3在推荐效果和作为时尚助手的实际价值方面的优越性能。 

---
