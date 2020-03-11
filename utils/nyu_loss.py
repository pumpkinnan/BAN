import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))


def discriminative_loss_single(prediction, correct_label, feature_dim, sem_label,
                               delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''

    ### Reshape so pixels are aligned along a vector
    # correct_label = tf.reshape(correct_label, [label_shape[1] * label_shape[0]])
    # 将prediction改变维度 改为N*5
    reshaped_pred = tf.reshape(prediction, [-1, feature_dim])
    k = tf.reduce_min(sem_label)


    ### Count instances
    # unique_labels是实例种类[1,2,3]就是有三个实例，unique_id是个correct_label等长的张量，用来显示correct_label中每一位
    # 属于哪个实例，counts每个实例中有多少点
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    # 数据类型转换
    counts = tf.cast(counts, tf.float32)
    # num_instances是实例个数
    num_instances = tf.size(unique_labels)

    # 将属于同一个实例的点所预测出来的特征向量相加，生成一个num_instances*5的矩阵
    segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

    # 用上述加和除以gt的每个实例的点的数量，得到一个新的num_instances*5的矩阵，是本次预测中每个实例中的所有点的特征向量的平均值
    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    # 将这个均值分配给每个实例中的点，如实例5的点的特征向量都是之前算的平均值
    mu_expand = tf.gather(mu, unique_id)

    ### Calculate l_var
    # distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    # tmp_distance = tf.subtract(reshaped_pred, mu_expand)
    # 下面两行是算reshaped_pred和mu_expand的距离的一范数，还是一个N*1的矩阵
    tmp_distance = reshaped_pred - mu_expand
    # 求行一范数
    distance = tf.norm(tmp_distance, ord=1, axis=1)

    distance = tf.subtract(distance, delta_v)
    # 将distance的数据值限制在0到distance
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    # 把GT中属于同一实例的点算出来的新的距离差加和
    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    if k == 0:
        l_var[0] = 0
    l_var = tf.div(l_var, counts)
    # 将l_var加和并变成一个值
    l_var = tf.reduce_sum(l_var)
    # 除实例个数
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    ### Calculate l_dist

    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3
    # 将原来的num_instances*5的矩阵，行扩展num_instances倍，列不变
    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    # 将原来的num_instances*5的矩阵，列扩展num_instances倍，行不变
    mu_band_rep = tf.tile(mu, [1, num_instances])
    # 将扩展后的矩阵变成num_instances * num_instances行5列的矩阵，矩阵没num_instances行是相同的，如前num_instances行相同
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

    # 做差
    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    # Filter out zeros from same cluster subtraction
    # 生成一个num_instances * num_instances的单位矩阵
    eye = tf.eye(num_instances)
    zero = tf.zeros(1, dtype=tf.float32)
    # 将单位矩阵变成布尔类型，1变成false，0变成true
    diff_cluster_mask = tf.equal(eye, zero)
    # 将矩阵变形成一个行向量(一维)
    diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])
    # tf.boolean_mask前一个矩阵保留后一个矩阵中为true的部分,此处是把全为零的行减去
    mu_diff_bool = tf.boolean_mask(mu_diff, diff_cluster_mask)

    # intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
    # zero_vector = tf.zeros(1, dtype=tf.float32)
    # bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    # mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    # 求行一范数
    mu_norm = tf.norm(mu_diff_bool, ord=1, axis=1)
    if k == 0:
        if num_instances > 1:
            for i in range(num_instances - 2):
                mu_norm[i] = 3.0
            for j in range(num_instances - 1):
                mu_norm[j+num_instances-1] = 3.0
    mu_norm = tf.subtract(2. * delta_d, mu_norm)
    # max(0,mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    def rt_0(): return 0.

    def rt_l_dist(): return l_dist

    l_dist = tf.cond(tf.equal(1, num_instances), rt_0, rt_l_dist)

    ### Calculate l_reg
    l_reg = tf.reduce_mean(tf.norm(mu, ord=1, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, sem_label,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss
    :return: discriminative loss and its three components
    '''

    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim, sem_label[i],
                                                                     delta_v, delta_d, param_var, param_dist, param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [correct_label,
                                                                                           prediction,
                                                                                           output_ta_loss,
                                                                                           output_ta_var,
                                                                                           output_ta_dist,
                                                                                           output_ta_reg,
                                                                                           0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def discriminative_loss_single_multicate(sem_label, prediction, correct_label, feature_dim,
                                         delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Discriminative loss for a single prediction/label pair.
    :param sem_label: semantic label
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''
    unique_sem_label, unique_id, counts = tf.unique_with_counts(sem_label)
    num_sems = tf.size(unique_sem_label)

    def cond(i, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg):
        return tf.less(i, num_sems)

    def body(i, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg):
        inds = tf.equal(i, unique_id)
        cur_pred = tf.boolean_mask(prediction, inds)
        cur_label = tf.boolean_mask(correct_label, inds)
        cur_discr_loss, cur_l_var, cur_l_dist, cur_l_reg = discriminative_loss_single(cur_pred, cur_label, feature_dim,
                                                                                      delta_v, delta_d, param_var,
                                                                                      param_dist, param_reg)
        out_loss = out_loss.write(i, cur_discr_loss)
        out_var = out_var.write(i, cur_l_var)
        out_dist = out_dist.write(i, cur_l_dist)
        out_reg = out_reg.write(i, cur_l_reg)

        return i + 1, ns, unique_id, pred, ins_label, out_loss, out_var, out_dist, out_reg

    output_ta_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    loop = [0, num_sems, unique_id, prediction, correct_label, output_ta_loss, output_ta_var, output_ta_dist,
            output_ta_reg]
    _, _, _, _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op = tf.while_loop(cond, body, loop)

    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_sum(out_loss_op)
    l_var = tf.reduce_sum(out_var_op)
    l_dist = tf.reduce_sum(out_dist_op)
    l_reg = tf.reduce_sum(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def discriminative_loss_multicate(sem_label, prediction, correct_label, feature_dim,
                                  delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss for multiple categories.
    :return: discriminative loss and its three components
    '''

    def cond(sem, label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(sem, label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single_multicate(sem_label[i], prediction[i],
                                                                               correct_label[i], feature_dim,
                                                                               delta_v, delta_d, param_var, param_dist,
                                                                               param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return sem, label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)

    _, _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [sem_label,
                                                                                              correct_label,
                                                                                              prediction,
                                                                                              output_ta_loss,
                                                                                              output_ta_var,
                                                                                              output_ta_dist,
                                                                                              output_ta_reg,
                                                                                              0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def new_loss(prediction, correct_label, feature_dim, point_xyz,
             alpha, beta):
    xyz = point_xyz[:, :3]
    xyz = tf.reshape(xyz, [-1, 3])
    reshape_pred = tf.reshape(prediction, [-1, feature_dim])

    unique_lables, unique_ids, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, dtype=tf.float32)
    num_instances = tf.size(unique_lables)

    # 将属于同一个实例的点所预测出来的特征向量相加，生成一个num_instances*5的矩阵
    segmented_sum = tf.unsorted_segment_sum(reshape_pred, unique_ids, num_instances)

    # 每个实例的特征向量的平均值
    s = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    # 将这个平均值赋给每一个实例中的点
    s_expand = tf.gather(s, unique_ids)

    tmp_distance = reshape_pred - s_expand
    distance = tf.norm(tmp_distance, ord=1, axis=1)
    distance = tf.subtract(distance, alpha)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    mu_sum = tf.unsorted_segment_sum(xyz, unique_ids, num_instances)
    mu = tf.div(mu_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_ids)

    tmp_spatial_distance = xyz - mu_expand
    spatial_distance = tf.norm(tmp_spatial_distance, ord=1, axis=1)
    spatial_distance = tf.reciprocal(1 + tf.exp(-spatial_distance))

    loss_intra = tf.unsorted_segment_sum(tf.multiply(spatial_distance, distance), unique_ids, num_instances)
    loss_intra = tf.norm(loss_intra, ord=1, axis=0)
    # num_instances = tf.to_float(num_instances)
    loss_intra = tf.div(loss_intra, tf.cast(num_instances, tf.float32))

    # 将原来的num_instances*5的矩阵，行扩展num_instances倍，列不变
    s_d = tf.tile(s, [num_instances, 1])
    s_b = tf.tile(s, [1, num_instances])
    s_b = tf.reshape(s_b, (num_instances * num_instances, feature_dim))

    s_diff = tf.subtract(s_b, s_d)

    eye = tf.eye(num_instances)
    zero = tf.zeros(1, dtype=tf.float32)
    diff_cluster_mask = tf.equal(eye, zero)
    diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])

    s_diff_bool = tf.boolean_mask(s_diff, diff_cluster_mask)

    s_norm = tf.norm(s_diff_bool, ord=1, axis=1)
    loss_inter = tf.subtract(beta, s_norm)
    # loss_inter = tf.square(beta - s_norm)
    loss_inter = tf.clip_by_value(loss_inter, 0., loss_inter)
    loss_inter = tf.square(loss_inter)
    # loss_inter = tf.norm(loss_inter, ord=1, axis=0)
    loss_inter = tf.reduce_mean(loss_inter)

    # 实例个数在一个块里可能只有一个，所以之前num_instances-1可能为0
    # loss_inter = tf.div(loss_inter, tf.cast(num_instances, tf.float32) * (tf.cast(num_instances, tf.float32) - 0.99))
    def rt_0(): return 0.

    def rt_l(): return loss_inter

    loss_inter = tf.cond(tf.equal(1, num_instances), rt_0, rt_l)

    loss = loss_intra + loss_inter
    return loss, loss_intra, loss_inter


def new_batch_loss(prediction, correct_label, feature_dim, point_xyz,
                   alpha, beta):
    def cond(label, batch, out_loss, out_loss_intra, out_loss_inter, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_loss_intra, out_loss_inter, i):
        loss, loss_intra, loss_inter = new_loss(prediction[i], correct_label[i], feature_dim,
                                                point_xyz[i], alpha, beta)

        out_loss = out_loss.write(i, loss)
        out_loss_intra = out_loss_intra.write(i, loss_intra)
        out_loss_inter = out_loss_inter.write(i, loss_inter)

        return label, batch, out_loss, out_loss_intra, out_loss_inter, i + 1

    out_ta_loss = tf.TensorArray(dtype=tf.float32,
                                 size=0,
                                 dynamic_size=True)
    out_ta_loss_intra = tf.TensorArray(dtype=tf.float32,
                                       size=0,
                                       dynamic_size=True)
    out_ta_loss_inter = tf.TensorArray(dtype=tf.float32,
                                       size=0,
                                       dynamic_size=True)

    _, _, out_loss_op, out_loss_intra_op, out_loss_inter_op, _ = tf.while_loop(cond, body, [correct_label,
                                                                                            prediction,
                                                                                            out_ta_loss,
                                                                                            out_ta_loss_intra,
                                                                                            out_ta_loss_inter,
                                                                                            0])
    sa_loss_op = out_loss_op.stack()
    sa_loss_intra_op = out_loss_intra_op.stack()
    sa_loss_inter_op = out_loss_inter_op.stack()

    sa_loss = tf.reduce_mean(sa_loss_op)
    sa_loss_intra = tf.reduce_mean(sa_loss_intra_op)
    sa_loss_inter = tf.reduce_mean(sa_loss_inter_op)

    return sa_loss, sa_loss_intra, sa_loss_inter


def new_loss_1(prediction, correct_label, feature_dim, point_xyz,
               alpha, beta, gama):
    xyz = point_xyz[:, :3]
    xyz = tf.reshape(xyz, [-1, 3])
    reshape_pred = tf.reshape(prediction, [-1, feature_dim])

    unique_lables, unique_ids, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, dtype=tf.float32)
    num_instances = tf.size(unique_lables)

    # 将属于同一个实例的点所预测出来的特征向量相加，生成一个num_instances*5的矩阵
    segmented_sum = tf.unsorted_segment_sum(reshape_pred, unique_ids, num_instances)

    # 每个实例的特征向量的平均值
    s = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    # 将这个平均值赋给每一个实例中的点
    s_expand = tf.gather(s, unique_ids)

    tmp_distance = reshape_pred - s_expand
    distance = tf.norm(tmp_distance, ord=1, axis=1)
    distance = tf.subtract(distance, alpha)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    mu_sum = tf.unsorted_segment_sum(xyz, unique_ids, num_instances)
    mu = tf.div(mu_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_ids)

    tmp_spatial_distance = xyz - mu_expand
    spatial_distance = tf.norm(tmp_spatial_distance, ord=1, axis=1)
    spatial_distance = tf.reciprocal(1 + tf.exp(-spatial_distance))

    loss_intra = tf.unsorted_segment_sum(tf.multiply(spatial_distance, distance), unique_ids, num_instances)
    loss_intra = tf.norm(loss_intra, ord=1, axis=0)
    # num_instances = tf.to_float(num_instances)
    loss_intra = tf.div(loss_intra, tf.cast(num_instances, tf.float32))

    # 将原来的num_instances*5的矩阵，行扩展num_instances倍，列不变
    s_d = tf.tile(s, [num_instances, 1])
    s_b = tf.tile(s, [1, num_instances])
    s_b = tf.reshape(s_b, (num_instances * num_instances, feature_dim))

    s_diff = tf.subtract(s_b, s_d)

    eye = tf.eye(num_instances)
    zero = tf.zeros(1, dtype=tf.float32)
    diff_cluster_mask = tf.equal(eye, zero)
    diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])

    s_diff_bool = tf.boolean_mask(s_diff, diff_cluster_mask)

    s_norm = tf.norm(s_diff_bool, ord=1, axis=1)
    loss_inter = tf.subtract(beta, s_norm)
    # loss_inter = tf.square(beta - s_norm)
    loss_inter = tf.clip_by_value(loss_inter, 0., loss_inter)
    loss_inter = tf.square(loss_inter)
    # loss_inter = tf.norm(loss_inter, ord=1, axis=0)
    loss_inter = tf.reduce_mean(loss_inter)

    # 实例个数在一个块里可能只有一个，所以之前num_instances-1可能为0
    # loss_inter = tf.div(loss_inter, tf.cast(num_instances, tf.float32) * (tf.cast(num_instances, tf.float32) - 0.99))
    def rt_0(): return 0.

    def rt_l(): return loss_inter

    loss_inter = tf.cond(tf.equal(1, num_instances), rt_0, rt_l)

    l_reg = tf.reduce_mean(tf.norm(s, ord=1, axis=1))

    loss = loss_intra + loss_inter + (gama * l_reg)
    return loss, loss_intra, loss_inter, l_reg










