def get_layer_name(model):
    names = []
    change_name =[]
    layer_num = []
    l=1
    for layer in model.layers:
        names.append(layer.name)
    for i in names:
        if i.find('conv2d')!=-1:
              change_name.append('conv2d')
              layer_num.append(l)
              l=l+1
        elif i.find('activation') !=-1:
              change_name.append('activation')
              layer_num.append(l)
              l=l+1
        elif i.find('max_pooling2d')!=-1:
              change_name.append('maxpooling2d')
              layer_num.append(l)
              l=l+1
        elif i.find('dropout')!=-1:
              change_name.append('dropout')
              layer_num.append(l)
              l=l+1
        elif i.find('dense')!=-1:
              change_name.append('dense')
              layer_num.append(l)
              l=l+1
        else:
              change_name.append(i)
              layer_num.append(l)
              l=l+1
    return names,change_name,layer_num

def get_layer_filters(model):
    fil = []
    fil_size =[]
    conv_count = 0
    filter_size = []
    conv_strides = []
    check = False
    count = False
    num_layer = []
    s = 3
    name = ''
    pool_fil_size =[]
    pool_num_layer=[]
    pool_strides=[]
    l=1
    for layer in model.layers:
        if layer.name.find('conv2d')!=-1 :
            fil.append(layer.filters) 
            num_layer.append(l)
            fil_size.append(layer.kernel_size)
            conv_strides.append(layer.strides)
            #conv_count +=1
        if layer.name.find('max_pooling')!=-1:
              pool_fil_size.append(layer.pool_size)
              pool_num_layer.append(l)
              pool_strides.append(layer.strides)
        l+=1
    
    return filter_size,fil,num_layer,pool_fil_size,pool_num_layer,pool_strides,conv_strides
    
def dense_layer(model):
    dense_count = 0
    dense_units = []
    dense_num = []
    dc = 1
    for layer in model.layers:
        if layer.name.find('dense')!=-1 :
            dense_num.append(dc)
            dense_units.append(layer.units) 
            dense_count +=1
        dc+=1
    return dense_count, dense_units, dense_num
      
class Theia(Callback):
    """Callback that terminates training when bug is encountered.
    """
    def __init__(self, inputs,inputs_test, batch_size,classes,input_type):
        super(Theia, self).__init__()
        self.inputs = inputs
        self.inputs_test = inputs_test
        
        self.classes = classes
        self.input_type = input_type
        self.fil = []
        self.counter = 0
        self.count=0
        self.check = False
        self.batch_size = batch_size
        
    def on_train_begin(self,logs=None):
        start_time = time.time()
        image_row = self.model.layers[0].input_shape[1]
        image_column = self.model.layers[0].input_shape[2]
        channel = self.model.layers[0].input_shape[3]
        
        
        if 'Dense' in str(self.model.layers[-1]):
          classes = self.model.layers[-1].units
        elif 'Dense' in str(self.model.layers[-2]):
          classes = self.model.layers[-2].units  
        c=0
        layer_name, change_name, layer_number = get_layer_name(self.model)
        layer_count = len(layer_name)
        dense_count, dense_units, dense_num = dense_layer(self.model)
        check,filter_size,fil,num_layer,s, pool_fil_size, pool_num_layer,pool_strides,conv_strides= get_layer_filters(self.model)
        
        
        message=[]
        mes=[]
        
      #Normalization of input
        if self.input_type == 1:
              min_train = np.min(self.inputs[0][0])
              max_train = np.max(self.inputs[0][0])
              min_test = np.min(self.inputs_test[0][0])
              max_test = np.max(self.inputs_test[0][0])
        elif self.input_type == 0:
              min_train = self.inputs.min()
              max_train = self.inputs.max()
              min_test = self.inputs_test.min()
              max_test = self.inputs_test.max()
        print(max_test)
        if  min_train < 0.0 or max_train > 1.0 :
              mes.append('Normalize the training data' )
              c+=1
        if  min_test < 0.0 or max_test > 1.0:
              mes.append('Normalize the test data' )
              c+=1

        #Inadequate batch size
        input_c =str(type(self.inputs))
        if 'DirectoryIterator' in input_c:
            sam = self.inputs.samples
        else:
            sam = len(self.inputs)
        
        if sam  >=20000:
        
                if self.batch_size >256:
                      mes.append('Decrease the batch size --> preferred 256 or less')
                      c+=1
        If sam < 20000
                if self.batch_size > 32:
                        mes.append('Decrease the batch size --> preferred 64 or less')
                        c+=1
                
              
        
      #   #Learning rate out-of-bounds
        if 'adadelta' in str(self.model.optimizer):
              if self.model.optimizer.learning_rate != 1.0:
                    mes.append('For Adadelta optimizer use learning rate 1.0')
                    c+=1
        elif self.model.optimizer.learning_rate >=0.0001 and  self.model.optimizer.learning_rate <= 0.01:
          pass
        elif self.model.optimizer.learning_rate > 0.01:
              mes.append('Decrease the learning rate --> preferred between 0.01 to 0.0001') 
              c+=1
        elif self.model.optimizer.learning_rate < 0.0001:
              mes.append('Increase the learning rate --> preferred between 0.01 to 0.0001')
              c+=1 
              
      #   #Non-saturating non-linearity
        i = 0
        d = 0
        act = 0
        sub = 'relu'
        activation=''
        for layer in self.model.layers:
            d += 1
            if i < (len(change_name)-1):
              if 'dense'  in change_name[i] or 'conv2d' in change_name [i]:
                  activation = str(layer.activation)
                  if 'activation' in change_name [i+1]:
                        act = 1
                  elif 'activation' not in change_name[i+1] and sub in activation:
                        pass
                  elif 'activation' not in change_name[i+1] and sub not in activation:
                        
                        message.append('Layer ' + str(layer_number[i]) +' : Add activation function --> preferred ReLU')
                        c+=1
              if 'activation' in change_name[i] and act == 1:  
                      if  sub not in (str(layer.activation)) :
                          message.append('Layer ' + str(layer_number[i]) +' : Change activation function --> preferred ReLU')
                          c+=1
                      elif activation not in sub and sub not in (str(layer.activation)) and d <= (layer_count-1):
                          message.append('Layer ' + str(layer_number[i]) + ' : Change activation function --> preferred ReLU and use activation once')
                          c+=1
            i+=1         
        
      #   #Mismatch between number of classes, last layer activation & loss function
        for layer in self.model.layers:
              if layer.name == layer_name[-1] :
                    if dense_units[-1] >= 2:
                          if 'softmax' in str(layer.activation):
                                if str(self.model.loss).find('categorical_crossentropy') !=-1:
                                  pass
                                else:
                                  mes.append('Change loss function --> Use categorical_crossentropy')
                                  c+=1

                          else:
                                if str(self.model.loss).find('categorical_crossentropy') !=-1:
                                  message.append('Layer ' + str(layer_number[-1]) +' : Change activation function --> Use softmax')
                                  c+=1
                                else:
                                  message.append('Layer ' + str(layer_number[-1]) +' : Change activation function --> Use softmax')
                                  mes.append('Change loss function --> Use categorical_crossentropy')
                                  c+=1
                    else:
                          if 'sigmoid' in str(layer.activation):
                                if str(self.model.loss).find('binary_crossentropy') !=-1:
                                  pass
                                else:
                                  mes.append('Change loss function --> Use binary_crossentropy')
                                  c+=1
                          else:
                                if str(self.model.loss).find('binary_crossentropy') !=-1:
                                  message.append('Layer ' + str(layer_number[-1]) +' : Change activation function --> Use sigmoid')
                                  c+=1
                                  
                                else:
                                  message.append('Layer ' + str(layer_number[-1]) +' : Change activation function --> Use sigmoid')
                                  mes.append('Change loss function --> Use binary_crossentropy')
                                  c+=1
                                  
      #   #Trade-off between convolution layer and fully connected layer
        co = 0
        de = 0
        den_layer = []
        if channel == 3:
            for l in range(len(change_name)):
                if 'conv2d' in change_name[l]:
                    co +=1
                    conv_layer = layer_number[l]
                elif 'dense' in change_name[l]:
                    de+=1 
                    den_layer.append(layer_number[l])
                
          
            if co < 3 and de >3:
                message.append('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                message.append('Layer '+ str(den_layer[-2]) + ' : Decrease number of hidden dense layers --> preferred one or two') 
                c+=1
            elif co < 3:
                #print('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                message.append('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                
                c+=1
            elif de >3:
                #print('Layer: '+ str(den_layer[-2]) + ' Decrease number of hidden dense layers preferred one or two')
                message.append('Layer '+ str(den_layer[-2]) + ' : Decrease number of hidden dense layers --> preferred one or two')
                c+=1
            
        if channel == 1:
             for l in range(len(change_name)):
                if 'conv2d' in change_name[l]:
                    co +=1
                    conv_layer = layer_number[l]
                elif 'dense' in change_name[l]:
                    de+=1 
                    den_layer.append(layer_number[l])
                    
             if co < 2 and de > 3:
                 message.append('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                 message.append('Layer '+ str(den_layer[-2]) + ' : Decrease number of hidden dense layers --> preferred one or two') 
                 c += 1
             elif co < 2:
                 message.append('Layer '+ str(conv_layer) +' : Increase number of convolution layers')
                 c+=1
               
             elif de>3:
                 
                message.append('Layer '+ str(den_layer[-2]) + ' : Decrease number of hidden dense layers --> preferred one or two')   
                c += 1 
         
      #Down-sampling with pooling
     
        kernel_size = []
        j=0
        i = 0
        for layer in self.model.layers:
              if 'conv2d' in change_name[i]:
                  kernel_size.append(layer.kernel_size)
                  j+=1
                  t = kernel_size[0]
                  #if len(kernel_size)>4 and kernel_size[0][0]==3:
                  print(len(kernel_size))
                  if len(kernel_size)>4:
                              message.append('Layer '+ str(j) + ' : Add pooling layer')
                              c+=1         
                              kernel_size = []
              elif 'maxpooling2d' in change_name[i]:
                        if kernel_size.count(t) == len(kernel_size) and len(kernel_size)>=2 and kernel_size[0][0] in range(5,15):
                              message.append('Layer '+ str(j) + ' : Add pooling layer')
                              c+=1
                        # elif len(kernel_size)>4 :
                        #      message.append('Layer '+ str(j-1) + ' : Add pooling layer')
                        #      c+=1
                        kernel_size = []
                        j+=1
              else:
                   kernel_size = []
                   j+=1 
              i+=1    
      #   #Incorrect pooling size
        for i in range(len(pool_fil_size)):
              if pool_fil_size[i][0] != pool_fil_size[i][1]:
                    message.append('Layer ' + str(pool_num_layer[i])+ ' : Use square pool size --> preferred (2,2) or (3,3)')
                    c+=1
              elif pool_fil_size[i][0] == pool_fil_size[i][1] and pool_fil_size[i][0] in [2,3]:
                    pass
              elif pool_strides[i][0] != pool_strides[i][1]:
                  message.append('Layer ' + str(pool_num_layer[i])+ ' : Use square pool strides --> preferred (2,2) or (3,3)') 
                  c+=1 
              elif pool_strides[i][0] == pool_strides[i][1] and pool_strides[i][0] in [2,3]:
                    pass
              else:
                message.append('Layer ' + str(pool_num_layer[i])+ ' : Decrease the pool strides --> preferred (2,2) or (3,3)')
                c+=1 

      #   #Global feature extraction
	
	layer_num = []
        test_list1 = dense_units[:]
        test_list1.sort(reverse = True)
        if test_list1 == dense_units:
                    flag = 1
        
        i = 0
        for j in dense_num[:-1]:
                if flag == 0:
                    mes.append('Layer ' + str(j)+ ' : Keep the units same or decrease units while going deeper')
                    c+=1
                elif units[i] < 64:
                    mes.append('Layer ' + str(j) + ' : Increase the units in dense layer --> preferred 64-1024') 
                    c+=1
                elif units[i] >1024:
                    mes.append('Layer ' + str(j) + ' : Decrease the units in dense layer --> preferred 64-1024') 
                    c+=1
                i+=1



       
                  
      
      # Local feature extraction
        ck = 0
        ch = 0
        for i in range(len(filter_size)):
              if filter_size[i] % 2 == 0:
                    message.append( 'Layer '+ str(num_layer[i])+ ' :  Odd dimension filters are preferred')
                    
                    ch = 1
        #a = sorted(message.items(), key=lambda x: x[1]) 
        
        if ch == 0:      
          f = 0
          kernel_size = []
          if image_row == image_column and image_row<=150:     
              if s == 1:
                    x=-1
                    j=0
                    for layer in self.model.layers:
                          if 'conv2d' in change_name[j]:
                              kernel_size.append(layer.kernel_size)
                              x+=1
                          if 'maxpooling2d' in change_name[j]:  
                                for i in kernel_size:
                                      if i[0]==i[1] and i[0]==3 and len(kernel_size)==1:
                                          message.append('Layer '+ str(num_layer[x]) + ' : Increase the receptive field --> Add more convolution layers with same configuration as of layer ' +str(num_layer[x]))
                                          c+=1
                                          kernel_size = [] 
                                break
                          j+=1
              elif s == 2:
                      x=0
                      for i in range(len(filter_size)):
                            if filter_size[i] in [3,5]:
                                x+=1  
                                pass
                            else:
                                message.append('Layer '+ str(num_layer[x])+' : Decrease the kernel size --> preferred (3,3) or (5,5) ')
                                ck=1    
              else:
                    for i in range(len(filter_size)):
                            if filter_size[i] in [3,5]:
                                f = 1
                            else:
                                message.append('Layer '+ str(num_layer[i])+' : Decrease the kernel size --> preferred (3,3) or (5,5) ')  
                                ck=1
                    if f == 1:
                        j=-1
                        for layer in self.model.layers:
                            if layer.name.find('conv2d')!=-1:
                              kernel_size.append(layer.kernel_size)
                              j+=1
                            if layer.name.find('max_pooling2d')!=-1:    
                              for i in kernel_size:
                                  if i[0]==i[1] and i[0]==3 and len(kernel_size)==1:
                                    message.append('Layer '+ str(num_layer[j]) + ' : Increase the receptive field by adding more convolution layers with same configuration as of layer ' +str(num_layer[j]))
                                    c+=1
                                    kernel_size = [] 
                                    break

          elif image_row == image_column and image_row > 150:   
              if s == 1:
                  kernel_size = []
                  j = 0
                  for layer in self.model.layers:
                      if layer.name.find('conv2d')!=-1:
                          kernel_size.append(layer.kernel_size)
                          conv_layer = layer.name
                          j+=1
                      if layer.name.find('max_pooling2d')!=-1: 
                          for i in kernel_size:
                              if i[0]==i[1] and i[0]==3 and len(kernel_size)==1:
                                  message.append('Layer '+ str(j) + ' : Increase the receptive field --> Add more convolution layers with same configuration as of layer ' +str(j))
                                  c+=1
                                  kernel_size = [] 
                                  break
                            
              elif s == 2:
                      for i in range(len(filter_size)):
                            if filter_size[i] in [3,5,7,9,11]:
                                pass
                            else:
                                message.append('Layer '+ str(num_layer[i])+' : Decrease the kernel size --> preferred (3,3) to (9,9)')  
                                ck=1
                                c+=1
                                break
        
        
        #Trade-off between depth of model and number of feature maps
        ck =0 
       
        if channel == 3:
            for i in range(len(fil)):
                if fil[i] in range(16,513):
                    pass
                elif fil[i] < 16:
                    message.append('Layer '+ str(num_layer[i]) +' : Increase the number of filters --> preferred between 16-512')
                    c+=1
                elif fil[i] > 512:
                    message.append('Layer '+ str(num_layer[i]) +' : Decrease the number of filters --> preferred between 16-512')
                    c+=1

       if channel == 1:
            
            for i in range(len(fil)):
                if fil[i] in range(6,256):
                    pass
                elif fil[i] < 6:
                    message.append('Layer '+ str(num_layer[i]) +' : Increase the number of filters --> preferred between 6-256')
                    c+=1
                elif fil[i] > 256:
                    message.append('Layer '+ str(num_layer[i]) +' : Decrease the number of filters --> preferred between 6-256')
                    c+=1

        res = sorted(message, key = lambda x: int(x.split()[1]))
        for i in res:
          print(i)
        res1= sorted(mes)
        for i in res1:
            print(i)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        if c>=1:
            self.model.stop_training = True
            sys.exit(1) 