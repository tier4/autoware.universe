import numpy as np
import os

def convert_model_to_csv(model,save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt(save_dir + "/weight_acc_layer_1.csv", model.acc_layer_1[0].weight.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/weight_steer_layer_1.csv", model.steer_layer_1[0].weight.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/weight_acc_layer_2.csv", model.acc_layer_2[0].weight.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/weight_steer_layer_2.csv", model.steer_layer_2[0].weight.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/weight_lstm_ih.csv", model.lstm.weight_ih_l0.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/weight_lstm_hh.csv", model.lstm.weight_hh_l0.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/weight_complimentary_layer.csv", model.complimentary_layer[0].weight.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/weight_linear_relu.csv", model.linear_relu[0].weight.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/weight_final_layer.csv", model.final_layer.weight.detach().numpy().astype(np.float64),delimiter=',')

    np.savetxt(save_dir + "/bias_acc_layer_1.csv",model.acc_layer_1[0].bias.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/bias_steer_layer_1.csv",model.steer_layer_1[0].bias.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/bias_acc_layer_2.csv",model.acc_layer_2[0].bias.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/bias_steer_layer_2.csv",model.steer_layer_2[0].bias.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/bias_lstm_ih.csv",model.lstm.bias_ih_l0.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/bias_lstm_hh.csv",model.lstm.bias_hh_l0.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/bias_complimentary_layer.csv",model.complimentary_layer[0].bias.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/bias_linear_relu.csv",model.linear_relu[0].bias.detach().numpy().astype(np.float64),delimiter=',')
    np.savetxt(save_dir + "/bias_final_layer.csv",model.final_layer.bias.detach().numpy().astype(np.float64),delimiter=',')
    vel_scale = np.zeros(2)
    vel_scale[0] = model.vel_scaling
    vel_scale[1] = model.vel_bias
    np.savetxt(save_dir + '/vel_scale.csv',vel_scale,delimiter=',')