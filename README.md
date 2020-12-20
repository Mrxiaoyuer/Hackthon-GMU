# Federated Learning system for Transporation Mode Prediction based on Personal Mobility Data

Personal mobility trajectories/traces could benefit a number of practical application scenarios, e.g., pandemic control, transportation system management, user analysis and product recommendation, etc. For example, 
Google COVID-19 Community Mobility Reports[1] demonstrated the daily people movement trend and have been utilized to accurately predict the influence of the community[2] like traveling agents, retail enterprises, etc. 
On the other hand, such mobility traces are also privacy-critical to the users, as they contain or can be used to infer highly private personal information, like home/work addresses, activity patterns, etc.
Therefore, how to effectively utilize the data with high privacy-preserving degree, as well as to benefit the real-world applications remains challengeable.

In this project, we propose to apply the novel Federated Learning (FL)[3] framework to address the transportation mode prediction task with the privacy preserving service-level requirement. 
   
As a deep-learning (DL) distributed training framework, Federated Learning could enable model training on the local devices without needs to upload the users' private data, thus greatly enhancing the privacy preserving capability as well as maintaining similar convergence accuracy of the model.
    Therefore, applying FL in the personal mobility data-related use scenarios have three major benefits:

###(1) High Privacy-Preserving Capability: The personal mobility data stays in the users' local devices and do not need to be sent to the central server, thus greatly reducing the risk of personal data leakage;

###(2) Implementation Efficiency: As there is no need to transmit the data to the central server, both the communication cost and the information transmission encryption efforts could be saved, thus achieving higher implementation efficiency;

###(3) Large-Scale User Participation: Meanwhile, the distributed training capability of FL enables salable amounts of users to flexibly participate in the training process, thus contributing and enhancing the overall application performance.

![alt text](https://github.com/Mrxiaoyuer/Hackthon-GMU/blob/main/system.jpg?raw=true)
