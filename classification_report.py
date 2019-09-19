import numpy as np 

class report:

    def __init__(self,actual, prediction):
        """
        Get Classification info
        
        """
        self.actual,self.prediction= actual, prediction
        self.confusion_matrix = self.__confusion_matrix()
    
    def accuracy(self):
        """Compute the Accuracy"""
        return np.mean(self.actual==self.prediction)
    
    def normed_cm(self):
        """
        Return Per Class Accuracy
        """
        n=  self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
        return np.round(n,decimals=2)
        
        
    def num_classes(self):
        """
        return the number of (unique) classes 
        """
        return len(np.unique(self.actual))
        
    def __confusion_matrix(self):    
        """
        Calculate the confusion matrix; labels are numpy array of classification labels
        """
        cm = np.zeros((self.num_classes(), self.num_classes()))
        for a, p in zip(self.actual, self.prediction):
            cm[a][p] += 1
        return cm

    def precision(self,average = None):
        """
        From confusion matrix, determine precision. 
        If average is selected choose option for micro or macro 
        """

        if average== 'micro':
            return np.sum(np.diag(self.confusion_matrix)) / np.sum(np.sum(self.confusion_matrix, axis = 0))
        elif average== 'macro':
            return np.average(np.diag(self.confusion_matrix)/ np.sum(self.confusion_matrix, axis = 0))
        return np.diag(self.confusion_matrix)/ np.sum(self.confusion_matrix, axis = 0)

    def recall(self,average = None):
        """
        From confusion matrix, determine recall. 
        If average is selected choose option for micro or macro 
        """
        if average== 'micro':
            return np.sum(np.diag(self.confusion_matrix)) / np.sum(np.sum(self.confusion_matrix, axis = 1))
        elif average== 'macro':
            return np.average(np.diag(self.confusion_matrix)/ np.sum(self.confusion_matrix, axis = 1))
        else: 
            return np.diag(self.confusion_matrix)/ np.sum(self.confusion_matrix, axis = 1)


    def f1(self,average = None):
        """
        From confusion matrix, determine f1 score. 
        If average is selected choose option for micro or macro 
        """

        if average== 'micro':
            p  = self.precision(average='micro')
            r = self.recall(average='micro')
        elif average== 'macro':
            p  = self.precision()
            r = self.recall()
            return np.average(np.nan_to_num( 2 * p * r / (p + r)))
        else:  
            p  = self.precision()
            r = self.recall()

        return np.nan_to_num(2 * p * r / (p + r))

    def create_report(self):
        """
        From actual and predicition values, create sklearn equivalent  
        of classification_report 
        """
        print ("Model Accuracy:\n------------------\n-- {:.2f}% --\n".format(self.accuracy()*100))
        print ("Confusion Matrix:\n------------------\n",self.confusion_matrix,"\n")
        print ("(Normalized) Confusion Matrix:\n------------------\n",self.normed_cm(),"\n")
        ind = np.arange(len(set(self.actual))).tolist()
        ind.extend(['','micro avg','macro avg'])
        p_list= self.precision().tolist()
        p_list.extend(['',self.precision('micro'),self.precision('macro')])
        r_list=self.recall().tolist()
        r_list.extend(['',self.recall('micro'),self.recall('macro')])
        f_list= self.f1().tolist()
        f_list.extend(['',self.f1('micro'),self.f1('macro')])
        
        print("Report:\n------------------\n")

        print(pd.DataFrame({"Precision:":p_list,
                      "Recall:":r_list,
                     "f1_score:":f_list},index=ind))
        
    def plot_cm(self, normalize=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = self.normed_cm()
            print("Normalized confusion matrix")
        else:
            cm =self.confusion_matrix
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(10,10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrx')
        plt.colorbar()
        tick_marks = np.arange(self.num_classes())
        classes=np.unique(self.actual).tolist()
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' 
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')