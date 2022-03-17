# Deep residual learning for image recognition

* **Authors** : Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
* **Journal** : ["arXiv preprint arXiv:1512.03385"](http://arxiv.org/abs/1512.03385)
* **Year** : 2015


## 问题的提出
* 深度网络的退化问题:
  * 深度卷积神经网络在图像分类领域取得了一系列的突破 。网络的深度对模型的性能至关重要，当增加网络层数后，网络可以进行更加复杂的特征模式的提取，所以当模型更深时理论上可以取得更好的结果。但是实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。这种退化并不是由过拟合造成的，并且在一个合理的深度模型中增加更多的层却导致了更高的错误率。
  * 退化的出现（训练准确率）表明了并非所有的系统都是很容易优化的。
## 解决方案(1)

* 残差学习:
  * 本文展示了一种残差学习框架，能够简化使那些非常深的网络的训练，该框架使得层能根据其输入来学习残差函数而非原始函数
* 设计思路:
  * 针对一个浅层的网络结构和一个相应的增添更多层后的深层的网络结构，如果新增加的层是一个恒等映射而前面的网络保持一致，那么这个深层的网络至少不会产生比浅层的网络更高的错误率。
  * 然而实验表明，当前的优化方法却不能让深的网络取得和浅的网络一样好或者更好的结果。也就是说，训练后的参数并不能达到上述想象中的效果，它无法使得新增添的层恰好成为了一个恒等映射层
* 残差学习:
  * 通过跳跃连接来实现了一个简单的恒等映射
  * 对于一个堆积层结构（几层堆积而成）当输入为 x 时其学习到的特征记为 H(x) ，现在我们希望其可以学习到残差 F(x)=H(x)-x ，这样其实原始的学习特征是 F(x)+x 。之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。
    <p align="center">
     <img src = "https://user-images.githubusercontent.com/98577810/158724998-cbdbe6b0-7dc4-41fe-ab27-dd029ddf31e2.png" width="300"/>
    </p>

  * 对比18-layer和34-layer的网络效果，如图7所示。可以看到普通的网络出现退化现象，但是ResNet很好的解决了退化问题。
    <p align="center">
      <img src = "https://user-images.githubusercontent.com/98577810/158728711-3fae97c0-dd52-4a72-9d29-dfca9ac7e216.png">
    </p>

  * ResNet网络与其他网络在ImageNet上的对比结果，如表所示。可以看到ResNet-152其误差降到了4.49%，当采用集成模型后，误差可以降到3.57%。
    <p align="center">
      <img src = "https://user-images.githubusercontent.com/98577810/158729438-2848cb63-b8fb-4983-9868-babbc4b6bbe0.png" width= "400"/>
    </p>
 
## 实验
* 我们要解决的问题是训练一个模型来对蚂蚁和蜜蜂进行分类 。我们为蚂蚁和蜜蜂提供了大约120张训练图像。每个类别有75个验证图像。
    <p align="center">
      <img src = "https://user-images.githubusercontent.com/98577810/158730050-e8b4b9b3-02fd-44e2-8dc4-5caa080abc31.png" width= "400"/>
    </p>


* 实验结果：辨别正确率达到了百分之93%
<table>
<!--   <tr>
    <td>First Screen Page</td>
     <td>Holiday Mention</td>
     <td>Present day in purple and selected day in pink</td>
  </tr> -->
  <tr>
    <td><img width="360" height="300" src="https://user-images.githubusercontent.com/98577810/158726646-b947a133-c5db-48ae-9e77-708972fc5e16.png"></td>
    <td><img width="360" height="300" src="https://user-images.githubusercontent.com/98577810/158726693-b6e2b33d-ca1d-47f8-8577-31e27bf3192d.png"></td>
    <td><img width="360" height="300" src="https://user-images.githubusercontent.com/98577810/158726710-2419bce3-977f-4bf0-88a0-0eee5b4a1cc6.png"></td>
  </tr>
 </table>

## 总结
* 相关工作：
  * 对于残差网络的研究，大部分集中在两个方向，第一个是结构方面的研究，另一个是残差网络原理的研究。
  * 更密集的跳层连接DenseNet：如果将ResNet的跳层结构发挥到极致，即每两层都相互连接，那么就成为了DenseNet，
  * 更多并行的跳层连接：如果将ResNet和DenseNet分为作为两个通道并行处理，之后再将信息融合，就可以得到Dual Path Network
  * 残差网络结构简化了学习过程，增强了梯度传播。打破了网络的不对称性。增强了网络的泛化能力



