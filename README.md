# pytorch0730
<br><br>



### 정규화 함수
----------

<br>
np.mean, np.std 로 직접 정규화를 해주는 함수에서
<br>
scikitlearn.preprocessing 의 StandardScaler 를 사용해 <br>
data의 평균을 1로, 분산을 0으로 만들어 정규화하였습니다.<br><br>

<table>
  <tr>
      <td> <br>0730
      </td>
      <td> <br>0729
      </td>
  </tr>
  <tr>
      <td valign="top" align="left">
          
~~~python

def dB( magnitude ):
    decibel = 20*np.log10( np.abs(magnitude) + np.finfo(float).eps )    
    return decibel


def scaler( L, R):
    LR = np.concatenate( (L,R), axis=0 )
    
    """normalization"""
    #z = MinMaxScaler().fit_transform(LR[:])
    z = StandardScaler().fit_transform(LR[:])
    #z = RobustScaler().fit_transform(LR[:])
    #z = MaxAbsScaler().fit_transform(LR[:])
    
    z = z.reshape(2, 257, 382)
    return z[0], z[1]
~~~
   </td>
   <td valign="top"> 
    
~~~python     
def dB( magnitude ):
    return 20*np.log10( np.abs(magnitude) + np.finfo(np.float32).eps )
    

def Mag_normalization( L, R ):

    Mag = np.asarray( [ L, R ] )  #(2, 257, 382)
    mu = np.mean( Mag )
    sigma = np.std( Mag )
    z = ( Mag - mu ) / sigma
    return z[0], z[1]


def Phase_normalization( phase ):
    mu = np.mean( phase )
    sigma = np.std( phase )
    
    z = ( phase - mu ) / sigma
    return z
~~~~
    
   </td>
  </tr>
</table>
    
<br><br><br>
    


### Screenshots
-------
epoch=100<br>
batch_size=20<br>
lr=0.00002<br>

<table>
  
  <tr> 
      <td colspan="4"><br><br> 0729 (np.mean, np.std) </td>
  </tr>

  <tr>
    <td> <img src="https://github.com/Kang-Dong-Hwi/pytorch0729/blob/master/Screenshots/train_dataset_confusion_matrix2905.png", height=230px, width=250px>  </td>
    <td> <img src="https://github.com/Kang-Dong-Hwi/pytorch0729/blob/master/Screenshots/validation_dataset_confusion_matrix2905.png", height=230px, width=250px>  </td>
    <td colspan="2"> <img src="https://github.com/Kang-Dong-Hwi/pytorch0729/blob/master/Screenshots/Adam2905.png", height=200px, width=350px>  </td>
  </tr>
  
  <tr> 
      <td colspan="4">
       training accuracy: 82%<br>
       validation accuracy: 11%<br>
      </td>
  </tr>
  <tr> 
      <td colspan="4"><br><br> 0730 (sklearn.preprocessing StandardScaler) </td>
  </tr>

  <tr>
    <td> <img src="https://github.com/Kang-Dong-Hwi/pytorch0730/blob/master/train_dataset_confusion_matrix3001.png", height=230px, width=250px>  </td>
    <td> <img src="https://github.com/Kang-Dong-Hwi/pytorch0730/blob/master/validation_dataset_confusion_matrix3001.png", height=230px, width=250px>  </td>
    <td colspan="2"> <img src="https://github.com/Kang-Dong-Hwi/pytorch0730/blob/master/Adam3001.png", height=200px, width=350px>  </td>
 </tr>
  
  <tr> 
      <td colspan="4">
       training accuracy: 91.250%<br>
       validation accuracy: 44.5%<br>
      </td>
  </tr>
</table>
