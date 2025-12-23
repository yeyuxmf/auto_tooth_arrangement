# Auto_tooth_arrangement(code: Model 1 application)
Implementation of traditional tooth alignment methods based on two reference papers(Appendix5 参考两篇传统排牙文献，实现传统牙齿排列方法)。

1.Automatically arrange teeth using neural networks. Applied to automatic orthodontic treatment of oral teeth.

2.This is only an automatic arrangement of single jaw teeth (considering certain factors). It is easy to expand to full mouth teeth.

3.This code is only for Model 1 and does not consider missing teeth. Reference paper for missing teeth, valid.

4.It does not involve the inevitable core technology in dental alignment.However, it is for academic research only and may not be used for commercial purposes without permission.

5.My purpose in establishing this git is to attract more contributions, lower the threshold(抛砖引玉，降低门槛), and hope that everyone can have different ideas. I hope to see new papers published with new ideas to promote automatic dental alignment in medicine.

6.During the training process, it is crucial to enable automatic augmentation of angles and translations to ensure that each training data set is unique. This is extremely important as it simulates the pre-orthodontic tooth posture infinitely.


# Reference:
1. TANet: Towards Fully Automatic Tooth Arrangement.
2. Tooth_Alignment_Network_Based_on_Landmark_Constraints_and_Hierarchical_Graph_Structure.
3. ViT：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
4. https://github.com/ziyangyeh/TANet-Pytorch.    (Thank you very much for the author's communication.)
5. https://github.com/facebookresearch/mae. (Mainly refer to the model structure section.)
6. https://github.com/liucongg/GPT2-NewsTitle.
7. https://github.com/jadore801120/attention-is-all-you-need-pytorch.
8. https://github.com/graykode/nlp-tutorial.
9. https://github.com/openai/gpt-3.

   
# model .pth
链接：https://pan.baidu.com/s/1nsectXx46bpWMqeVSbYnTQ 
提取码：0chs    
The reason why the model file is so large is because I did not delete some useless parameters.
"I apologize for the delay; the model has been lost from the cloud storage. 
# Visualization Tool Installation Package
Download geomagic2013 online and install it yourself.
This is a 3D visualization tool that supports visualization of stl files and point cloud txt file formats.

# Model structure
![auto teeth_model](https://github.com/huang229/auto_tooth_arrangement/assets/29627190/bb9f0579-2108-4f08-bb86-ac30cb37709d)


![loss](https://github.com/huang229/auto_tooth_arrangement/assets/29627190/21831ebe-0bb4-48ec-9ef2-742ac1c5d3fe)



![Big Data Training](https://github.com/huang229/auto_tooth_arrangement/assets/29627190/016d618f-8416-4a70-bd6e-a555b7eaaec3)


# The reason for this design：
1. Tooth arrangement includes two operations: translation and rotation.
2. The model should have modules that care about rotation and translation separately.
3. If you only care about rotation, then you should not be affected by position. So we should decentralize every tooth.
4. If we only care about translation, then we should not be affected by tooth posture. The center point of the tooth is undoubtedly the characteristic that best reflects the position of the tooth.
5. Embed tooth position information to better reflect the relative position relationship of each tooth. When you only train the tooth translation, the value of the embedded position information Loss function decreases less.
6. Residual module, you don't know the global information required for each tooth, so set a learnable parameter weight for each tooth.
7. Pay attention to the direction of the teeth, so the input should not be a feature unrelated to the tooth posture. Therefore, a transformer model is used here, which can extract the global features of each tooth.
8. The first transformer module calculates the relationship between teeth in feature extraction of tooth center points, as the continuity of teeth on the dental arch is essentially the relative pose relationship between teeth. You can also use a fully connected layer instead, as the third transformer module also calculates the correlation between teeth.
9. The fusion of center point features and tooth features is because tooth arrangement is a relative pose relationship of teeth on the dental arch. For example, even if the position is correct but the posture is not good, serious collisions may still occur, and poor posture can also affect bite, and so on.
10. Finally, separate the prediction of tooth translation and rotation, and consider that translation and rotation are two different tasks.

    
# Environment
1.python 3.7.0

2.pytorch 11.3.1

3.pytorch3D 0.7.4    

# Train  
python main.py

1.Replace the optimizer. opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

2.When using your own data, preprocessing may vary, and you need to calculate the values of your rotation and translation matrices to obtain gdofs and gtrans.

3.If there are more than 300 dental data samples, it is recommended to sample each sample approximately 10 times. If the dental data is less than 300, it is recommended to sample 10-20 times for each data. Because this is random sampling and has non-uniformity, using farthest point sampling is too time-consuming and not advisable.

4.The initial training suggestion is to only turn on reconstruction, angle, and translation loss. After the model converges to the minimum, fine tuning can be performed to turn on the metric_ Loss and Spatial_ Relationship_ Loss function (loss of center symmetry of left and right teeth and spatial relationship between adjacent teeth). Adjust the weights according to your needs.
   
# test
python test_rotate.py


# Difficult issues:
1.Orthodontics: The collision between teeth is undoubtedly a crucial point, and what constraints should be established to 100% avoid tooth collisions in the dental alignment results. The occlusion and coverage between teeth are all worth in-depth consideration.

2.I hope to have like-minded friends to discuss together. Leave QQ or wechat for easy communication.


# Appendix1
1.When doctors perform dental correction on patients, not all teeth will change, but only a few teeth or some of their positions will change. For example, if the patient only has a deformed position of the incisors, they may only move and rotate the incisors, or move and rotate them together with the lateral incisors and canines near the incisors. The posterior premolars and molars may remain unchanged during orthodontic treatment, so the design of Model 1 is unreasonable. Doctors often refer to the position of dental arches and teeth without deformities when performing orthodontic treatment on patients. Therefore, it can be understood as only adjusting the relative posture of some deformed teeth based on referencing the correct tooth posture, or it may be a back and forth dynamic adjustment process (for deformed teeth and adjacent teeth).Just like the language model in NLP, using the features obtained by the encoder and the predicted words to predict the next word.
医生对患者进行牙齿矫正，并不是所有的牙齿都会变化，只是少数的几颗牙齿或者部分牙齿的位姿发生变化。比如患者只有切牙的位姿畸形，那么可能只移动和旋转切牙，或者连同切牙附近的侧切牙和尖牙一起移动和旋转，后面的前磨牙和磨牙在正畸治疗过程中可能是毫无变化的，因此模型1设计不合理。医生对患者进行牙齿正畸治疗，往往参考了牙弓和没有畸形的牙齿的位姿等等。因此可以理解为在参考正确的牙齿位姿的基础上仅对部分畸形牙齿做相对位姿调整，也可能是来回动态调整过程（那畸形牙及邻牙）。就像NLP中的语言模型一样，使用encoder得到的特征和已预测的单词去预测下一个单词。

1.model structure:
![model 2](https://github.com/huang229/auto_tooth_arrangement/assets/29627190/ff172d69-94cd-4274-8812-862ae392d34c)


Note: After the experiment, the model testing effective, the implementation of the gt part was hidden(Not convenient for discussion, you need to design it yourself).
The final loss value on big data is as follows:
![loss value](https://github.com/huang229/auto_tooth_arrangement/assets/29627190/238a010d-b587-4ccb-aa46-ef41516e29d8)


There is also a simple design. The doctor's orthodontic treatment as understood above only applies pose changes to some deformed teeth. Therefore, it is possible to consider changing the loss on Model 1 by adding a mask to calculate the loss value only for teeth with pose changes, or the weight of the loss value for teeth with unchanged pose is very small. It does not need to be a prediction of teeth that require pose changes like Model 2 and Model 3.还有一种简单设计。上面理解的医生正畸治疗仅对部分畸形牙齿做位姿变化，因此，可以考虑在模型1上对损失做变化，通过添加mask，仅对有位姿变化的牙齿计算损值，或者位姿不变的牙齿的损失值权重非常小。它就不必像模型2和模型3那样是一颗一颗的预测需要进行位姿变化的牙齿。
![mask loss](https://github.com/huang229/auto_tooth_arrangement/assets/29627190/41606c39-1cf6-44c2-a1c6-705782577fb3)



2.Collision issues：
By voxelizing the mesh, it can effectively represent the collision overlap between teeth, such as minimizing the loss value in the overlapping area.
Reference paper: 
  a.Mesh R-CNN https://arxiv.org/pdf/1906.02739.pdf
  b.Object Rearrangement Using Learned Implicit Collision Functions

You can also directly refer to the collision loss of meshes(my other project). https://github.com/huang229/mesh_collision_loss

For automatic orthodontic treatment, collision is a key core technical issue that is too important, and this part of the model structure design will no longer be displayed. Those who are interested can imagine for themselves. This is also an issue that has been avoided in published papers.

3.According to the assumptions of Model 2 and Model 3, using reinforcement learning to predict the position of the next deformed tooth would be a perfect design. Because as mentioned above, orthodontics is a process in which doctors dynamically adjust deformed teeth back and forth (repeatedly).(按照模型2和模型3的设想，如果在预测下一颗畸形牙齿的位姿时使用强化学习，将是完美的设计。因为上面提到牙齿正畸更是医生对畸形牙齿来回(反复)动态调整的过程。)

# Appendix2

1. The situation of only some teeth moving and rotating during orthodontic treatment was discussed in the above Appendix1, that is to say, before orthodontic treatment, it is known which teeth will have position and posture changes and which teeth will not. Based on this consideration, can a learning constraint be established? The answer is yes, and this consideration will greatly improve the tooth arrangement effect. The following advantages: 1. It will minimize the collision after tooth arrangement; 2. The tooth arrangement result has a high degree of overlap with the doctor's tooth arrangement result.(上面附加1中讨论了牙齿正畸时仅部分牙齿移动和旋转的情况，也就是说正畸前就知道哪些牙齿会有位置和姿态变化，哪些牙齿没有，基于这个考虑能否建立一种学习约束，答案是肯定的，而且这种考虑会大幅度提升排牙效果，如下优势：1.它会尽可能减少排牙后的碰撞；2.排牙结果与医生排牙结果有高度的重叠。)
![png](https://github.com/user-attachments/assets/4eaa01e9-e3d3-43d4-b61b-a574d47f16d0)


2. The same patient may receive different treatment plans from different doctors, resulting in different arch forms (such as nature arch, tappered arch, oval arch, and square arch) after orthodontics. At the same time, different types of dental arch tendencies may also receive different orthodontic results. In other words, different doctors may achieve diverse orthodontic results on the premise of medical compliance.(同一位患者，不同医生有不同的治疗方案，因此正畸后牙齿会呈现出不同的弓形（比如：nature arch, Tappered arch、Ovoid arch  and Square arch)，同时，不同医生倾向同一种弓形也会得到不同的正畸结果，也就是说不同医生会以符合医学为前提得到多样性的正畸结果。)
![7](https://github.com/user-attachments/assets/8c939df2-93ad-4d29-9f49-87022a9a70a2)

The above figure shows the overlap difference of different arch outputs.
Here, the same tooth is displayed using a model to obtain five different arch outputs. In addition, it is easier to obtain different results for the same arch, which are not shown.

Note: Additional 2 does not show any code or principle, and requires readers to think for themselves.(附加2不展示任何代码和原理，需读者自己思考。)

# Appendix3
The reinforcement learning mentioned in Appendix1 is feasible and has been verified.
Two design ideas: 1. Consider a single agent model, such as a robotic arm assembling objects in sequence; 2. Multi-agent linkage model, such as the 5vs5 game of League of Legends or Dota.

# Appendix4
It has been more than a year. This is the integration of automatic tooth arrangement, and the model supports automatic alignment of any number of missing teeth (up to 32 teeth).

This is the effect over a year ago, with only renderings, but no open-source code. This can be seen as a response to those who have doubts about whether this open-source project can be extended to align the entire jaw of teeth. I think the key lies in whether one understands the transformer, a deep learning module. If one understands it, then they can expand it; if not, they may not be able to complete the task.

The effect is as follows, This should be a mediocre rendering, and it's the only ready-made picture I have.
![model arrangement](https://github.com/user-attachments/assets/27ca9e1a-7300-4dc1-895f-a6aaffa11da8)

# Medical constraints
1. Each tooth has its own medical feature points, such as neighboring points, tangential points, cusps, and so on. These feature points all have neighboring or contact relationships, which can minimize their distance.

2. The tooth arrangement should not detach from the alveolar bone. Can we calculate a dental arch curve based on the alveolar bone and tooth feature points, and then sample the dental arch curve to minimize the distance between a certain feature point of the tooth and the sampling point.




# Appendix5

Note: This is only a reference to the methods in the papers, not a reproduction.
My capabilities are limited, for reference only.(水平有限，仅供参考)

5.1. Reference:
   
1.Li H, Liu M. An automatic teeth arrangement method based on an intelligent optimization algorithm and the Frenet–Serret formula[J]. Biomedical Signal Processing and Control, 2025, 105: 107606.

2.Ping Y, Wei G, Wei G, et al. A Rule-Based Optimization Method for Tooth Alignment[J]. IEEE Transactions on Visualization and Computer Graphics, 2025.

5.2. Application

1. Traditional_Multi-Objective_Particle_Swarm_Optimization(MOPSO).py  (Reference 1)

2. Traditional_energy_minimization_arrangement(lbgfs).py   (Reference 2)

3. Traditional_tooth_mesh_collsion(c++vs2017)  (Implementing mesh-to-mesh collision detection between two teeth in C++.)(eigen-3.4.0   OpenMesh 9.1)

5.3. sansas Multi-Objective Optimization Energy Funct
1. Curve Attachment Energy

    Logic: Calculates the minimum Euclidean distance between the tooth’s current center (FA point) and the fitted arch curve.

    Objective: To ensure all teeth are aligned along the predefined arch trajectory, preventing them from deviating from the intended track.

2. Midline Symmetry Energy

    Logic: Specifically targets the central incisors (teeth 8 & 9) and lateral incisors (teeth 7 & 10) by calculating the difference in distance to the dental midline between symmetrical pairs.

    Objective: To maintain esthetic symmetry, ensuring the left and right anterior teeth are balanced relative to the facial midline.

3. Position Persistence Energy (Displacement Constraint)

    Logic: Measures the average Euclidean distance between the current tooth position and its initial sampled position ($s_0$).

    Objective: To act as an "anchor," preventing the optimizer from making drastic, unstable leaps in tooth position while pursuing other objectives, thus ensuring a smooth optimization process.

4. Ideal Distribution Energy (Spacing)

    Logic: Compares the current parametric position ($s_c$) of a tooth on the arch curve with its ideal proportional position ($s_{ideal}$) derived from the tooth numbering.

    Objective: To ensure even tooth distribution along the arch; for example, ensuring tooth 2 is near the start and tooth 15 is near the end, preventing localized crowding or excessive gaps.

5. Hard Collision Penalty

    Logic: Invokes the C++ collision engine to apply a massive energy penalty when a penetration between two tooth meshes is detected (i.e., signed distance $d_{min} < 0$).

    Objective: To enforce physical constraints, preventing "interpenetration" between 3D dental models, which is a fundamental requirement for a valid dental setup.

6. Proximal Contact Energy (Gap Goal)

    Logic: For adjacent teeth, it calculates the deviation between their actual minimum gap and the target contact distance ($margin\_max = 0.1$).

    Objective: To ensure adjacent teeth are neither colliding nor separated by gaps, simulating tight proximal contact (i.e., "closing spaces" in orthodontics).

7. Axial Alignment Energy

    Logic: Uses get_rot_vec to calculate the angular deviation between the tooth's intrinsic mesiodistal vector and the local tangent of the arch curve.

    Objective: To ensure the "long axis" or mesiodistal orientation of each tooth follows the curvature of the arch, resulting in a neat, well-oriented alignment rather than random rotations.

5.4. Result

<img width="1247" height="899" alt="lbgfs_tooth_arrangement" src="https://github.com/user-attachments/assets/16874b2a-fd06-4989-aa57-5e0a7c0881c5" />

1.Left: Pre-alignment state; the arch tangent and the tooth tangent are not coincident. Right: Post-alignment state; the arch tangent and the tooth tangent are perfectly coincident.


5.5. Give up on collision detection(f5 and f6 ==0)

<img width="600" height="435" alt="1" src="https://github.com/user-attachments/assets/1d43e438-e882-403a-9f70-f767d1a52150" />


5.6. Discussion

1. The introduction of collision terms ($f_5$ and $f_6$) significantly interferes with the minimization of other energy functions ($f_1, f_2, f_3, f_4, f_7$), frequently leading to optimization collapse.
引入碰撞后(f5和f6)，对其它能量函数(f1,f2,f3,f4,f7)的最小化非常明显，优化很容易崩溃。

2. I propose to first generate the dental arch curve independently. Then, starting from the arch center and moving toward both sides, the teeth are sequentially arranged along the curve based on acceptable collision gaps. Simultaneously, the tangent of each tooth is rotated to align with the local tangent of the arch curve. Finally, the previously mentioned energy minimization is applied for fine-tuning. This strategy provides a high-quality initial configuration, effectively resolving the optimization collapse caused by the introduction of collision constraints.
我认为先直接只算出牙弓曲线，然后牙齿依据可接受的碰撞间隙把牙弓从牙弓中心往两边一次排开，且同时牙齿切线旋转为牙弓切线方向；然后再使用上面的能量最小化微调牙齿排列，这样可以解决碰撞的引入造成的优化崩溃问题。

3. Beyond the FA points and mesiodistal points used for tangent calculation, more feature points should be introduced to directly establish the tooth's Local Coordinate System (LCS). This would allow for a direct transformation (rotation and translation) of the tooth into the dental arch's local frame, resulting in a highly smooth initial arrangement. Following this rigid alignment, the energy minimization process can be applied to further optimize the placement.
除了牙齿的FA点和计算切线中的近远中点，应该引入更多的特征点，比如直接获得牙齿的局部坐标系，这样可以把牙齿直接旋转到牙弓坐标系上，得到的排列结果就非常光滑。然后再根据能量最小化优化。

4. If the mesiodistal width of each tooth is known, the proportional constraints for the equidistant points in $f_4$ can be calculated based on tooth width ratios. This approach yields significantly better constraint performance and a more realistic distribution.
如果已知每颗牙齿的切线方向宽度，那么利用牙宽比例计算F4中的等分点，这样的约束效果更好。



# License and Citation
1. Without permission, the design concept of this model shall not be used for commercial purposes, profit seeking, etc.
2. If you refer to the design concept of this model for theoretical research and publication of papers on automatic tooth arrangement, please also add a reference(to git).
参考了，写论文请做出说明和添加引用。
