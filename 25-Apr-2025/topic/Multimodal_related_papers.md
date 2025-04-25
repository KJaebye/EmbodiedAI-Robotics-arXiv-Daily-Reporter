# Hierarchical and Multimodal Data for Daily Activity Understanding 

**Title (ZH)**: 层级化和多模态数据在日常活动理解中的应用 

**Authors**: Ghazal Kaviani, Yavuz Yarici, Seulgi Kim, Mohit Prabhushankar, Ghassan AlRegib, Mashhour Solh, Ameya Patil  

**Link**: [PDF](https://arxiv.org/pdf/2504.17696)  

**Abstract**: Daily Activity Recordings for Artificial Intelligence (DARai, pronounced "Dahr-ree") is a multimodal, hierarchically annotated dataset constructed to understand human activities in real-world settings. DARai consists of continuous scripted and unscripted recordings of 50 participants in 10 different environments, totaling over 200 hours of data from 20 sensors including multiple camera views, depth and radar sensors, wearable inertial measurement units (IMUs), electromyography (EMG), insole pressure sensors, biomonitor sensors, and gaze tracker.
To capture the complexity in human activities, DARai is annotated at three levels of hierarchy: (i) high-level activities (L1) that are independent tasks, (ii) lower-level actions (L2) that are patterns shared between activities, and (iii) fine-grained procedures (L3) that detail the exact execution steps for actions. The dataset annotations and recordings are designed so that 22.7% of L2 actions are shared between L1 activities and 14.2% of L3 procedures are shared between L2 actions. The overlap and unscripted nature of DARai allows counterfactual activities in the dataset.
Experiments with various machine learning models showcase the value of DARai in uncovering important challenges in human-centered applications. Specifically, we conduct unimodal and multimodal sensor fusion experiments for recognition, temporal localization, and future action anticipation across all hierarchical annotation levels. To highlight the limitations of individual sensors, we also conduct domain-variant experiments that are enabled by DARai's multi-sensor and counterfactual activity design setup.
The code, documentation, and dataset are available at the dedicated DARai website: this https URL 

**Abstract (ZH)**: 基于人工 Intelligence 的日常活动记录 (DARai) 

---
# MCAF: Efficient Agent-based Video Understanding Framework through Multimodal Coarse-to-Fine Attention Focusing 

**Title (ZH)**: MCAF：通过多模态粗细注意力聚焦的高效基于代理的视频理解框架 

**Authors**: Shiwen Cao, Zhaoxing Zhang, Junming Jiao, Juyi Qiao, Guowen Song, Rong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17213)  

**Abstract**: Even in the era of rapid advances in large models, video understanding, particularly long videos, remains highly challenging. Compared with textual or image-based information, videos commonly contain more information with redundancy, requiring large models to strategically allocate attention at a global level for accurate comprehension. To address this, we propose MCAF, an agent-based, training-free framework perform video understanding through Multimodal Coarse-to-fine Attention Focusing. The key innovation lies in its ability to sense and prioritize segments of the video that are highly relevant to the understanding task. First, MCAF hierarchically concentrates on highly relevant frames through multimodal information, enhancing the correlation between the acquired contextual information and the query. Second, it employs a dilated temporal expansion mechanism to mitigate the risk of missing crucial details when extracting information from these concentrated frames. In addition, our framework incorporates a self-reflection mechanism utilizing the confidence level of the model's responses as feedback. By iteratively applying these two creative focusing strategies, it adaptively adjusts attention to capture highly query-connected context and thus improves response accuracy. MCAF outperforms comparable state-of-the-art methods on average. On the EgoSchema dataset, it achieves a remarkable 5% performance gain over the leading approach. Meanwhile, on Next-QA and IntentQA datasets, it outperforms the current state-of-the-art standard by 0.2% and 0.3% respectively. On the Video-MME dataset, which features videos averaging nearly an hour in length, MCAF also outperforms other agent-based methods. 

**Abstract (ZH)**: Even in the Era of Rapid Advances in Large Models, Video Understanding, Particularly Long Videos, Remains Highly Challenging: MCAF,一种基于代理的无需训练框架，通过多模态粗细注意力聚焦进行视频理解 

---
