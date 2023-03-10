#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:33:55 2023

@author: vog20246
"""

torch.utils.data.DataLoader(train_db,
                            batch_size=10,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
'test':