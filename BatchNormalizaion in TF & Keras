TF1.x
tf.layers.batch_normalization
* training  
    * default=False 即使用滑动平均的均值和方差 | 滑动平均的均值、方差初始为0和1，所以如果从没打开过那BN不生效
    * 作用一：控制normalize是使用当前batch的统计量还是滑动平均的统计量
    * 作用二：为True的时候才会把moving_xx两个ops加到 tf.GraphKeys.UPDATE_OPS 里
    * Whether to return the output in training mode (normalized with statistics of the current batch) or in inference mode (normalized with moving statistics)
* trainable 
    * default=True 即gamma、beta放到TRAINABLE_VAR..这个collection里
    * if `True` also add variables to the graph collection `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).

一个正常的BN：
* training
训练的时候training=True(把两个moving放到UPDATE_OPS里)，且执行update_ops;
测试的时候training=False即使用训练集的统计量来进行预测；
* trainable
一般不管他，都是用true；


TF2.x
针对BN和dropout个人看法如下：
* 如果使用Sequential+fit() 那万事大吉；
* 如果使用Subclassed+fit() 需要在定义SubclassedModel类时，给此class的call()方法加上training参数 `def call(self,inp,training=None)`
* 如果使用Subclassed/Sequential+custom_loop 自定义训练loop，就需要在model(inp)时加上 `model(inp,training=False)`

tf2.0里的行为
[0] 总结：
* call方式调用需要指定trainning参数，model(inp, trainning=True)
* fit方式由trainable属性控制, model.featM.trainable=True model.fit(inp)
[1] .trainable=True/False时 以call来调用如果不给training参数，mean、variance均不会重新计算 
[2] model.fit是受trainable影响的，.trainable=True时 可以改变mean、variance， Flase时则不会
所以只训练FC的时候，如果设置了featM.trainable=False, 然后对FC做warmup/transfer-learning是可行的（bn的行为在训练/预测阶段都是一致的）
在fine-tuning中BN层的均值方差都是重新计算过的，并且预测阶段使用的也是计算后的均值方差
[3] tf2.0.0中tf.keras.backend.set_learning_phase=1 不生效
[4] 一个针对 「InceptionV3(weights='imagenet',include_top=False)」 直接加载模型的测试
结果基本上这个模型只能用custom loop然后手动给call加上training=False参数（.fit不行主要是因为tf2.0.0这里set_learning_phase失效了


keras实现上的争议：主要争议在于，设置trainable=False时，BN层要不要把mean和variance也冻结住 | 参见此PR(https://github.com/keras-team/keras/pull/9965)
"datumbox观察到 trainable=False时 mean、variance还是会更新，fchollet认为应该用set_learning_phase而不是trainable来控制"
但是使用set_learning_phase的问题有二，首先是不直观（当然这可以说是"trainable"和"fronzen"之间理解的区别）;
此外注意已经compile过的模型 set_learning_phase 是不会生效的（比如迁移学习想使用 tf.keras.applications.inception_v3.InceptionV3加载模型，必须要先设置learning_phase=True，再加载初始化此模型）


