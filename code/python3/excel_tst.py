# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 00:17:46 2018

@author: enggh
"""
from pandas import DataFrame

if __name__ == '__main__':
    val_d_dim = [20,20,20,50,50,50,100,100,100]
    val_m_size = [10,20,30,10,20,30,10,20,30]
    val_best_epochs = [10,20,30,10,20,30,10,20,30]
    val_auc_acc = [10,20,30,10,20,30,10,20,30]
    val_accuracy= [10,20,30,10,20,30,10,20,30]
    val_loss = [10,20,30,10,20,30,10,20,30]

    for i in range(0,9):
        df = DataFrame({'State Dimension':val_d_dim[i],'Memory Size':val_m_size[i],'The Best Epoch': val_best_epochs[i], 'Test AUC': val_auc_acc[i], 'Test Accuracy':val_accuracy[i],'Test Loss':val_loss})
    df.to_excel('Model_Validation.xlsx', sheet_name='DKVMN', index=False)
    

