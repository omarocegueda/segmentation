cimport cython
import numpy as np
cimport numpy as np

##----------------------------------------------------------------------------------
##----------------------------------------------------------------------------------

#
# Function to evaluate the DICE coefficient.
# Input:  3D numpy integer array data with the segmentation map,
#         3D numpy integer array data_ref with the gold standard reference,
#         class labels list of strings class_labels with the string labels
#         of each brain tissue, class label integer ids class_label_ids with
#         the integer id corresponding to each class_labels element (must be
#         in the same order of appearance and valid ids are >0), number of 
#         classes nclasses (must be 3 or 4), name of the file to write the
#         results file_name, boolean flag append set to >0 if the writing on
#         the file is to be appended or 0 in other case and total segmentation
#         time final_time in seconds.
# Output: Nothing. Writes the evaluation result to the file_name file in
#         either append mode or in write mode.
@cython.boundscheck(False)
def DICE(np.ndarray[np.int64_t, ndim=3] data, np.ndarray[np.int64_t, ndim=3] data_ref, \
         class_labels, np.ndarray[np.int64_t, ndim=1] class_labels_ids, int nclasses, \
         str file_name, int append, double final_time) :
   cdef int i,j,k,i1,dim1,dim2,dim3,segValue,gtValue
   nume      = np.array(np.zeros(nclasses),dtype=np.double)
   deno      = np.array(np.zeros(nclasses),dtype=np.double)
   deno_ref  = np.array(np.zeros(nclasses),dtype=np.double)
   DICE_coef = np.array(np.zeros(nclasses),dtype=np.double)
   dim1 = data.shape[0]
   dim2 = data.shape[1]
   dim3 = data.shape[2]
   if nclasses == 4 :
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
          if data_ref[i,j,k] <> 10 :
             segValue = data[i,j,k]
             gtValue  = data_ref[i,j,k]
             if segValue == class_labels_ids[0] :
                deno[0] = deno[0] + 1
             elif segValue == class_labels_ids[1] :
                deno[1] = deno[1] + 1
             elif segValue == class_labels_ids[2] :
                deno[2] = deno[2] + 1
             else :
                if segValue == class_labels_ids[3] :
                   deno[3] = deno[3] + 1
             if gtValue == class_labels_ids[0] :
                deno_ref[0] = deno_ref[0] + 1
                if segValue == class_labels_ids[0] :
                   nume[0] = nume[0] + 1
             elif gtValue == class_labels_ids[1] :
                deno_ref[1] = deno_ref[1] + 1
                if segValue == class_labels_ids[1] :
                   nume[1] = nume[1] + 1
             elif gtValue == class_labels_ids[2] :
                deno_ref[2] = deno_ref[2] + 1
                if segValue == class_labels_ids[2] :
                   nume[2] = nume[2] + 1
             else :
                if gtValue == class_labels_ids[3] :
                   deno_ref[3] = deno_ref[3] + 1
                   if segValue == class_labels_ids[3] :
                      nume[3] = nume[3] + 1
   elif nclasses == 3 :
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
          if data_ref[i,j,k] <> 10 :
             segValue = data[i,j,k]
             gtValue  = data_ref[i,j,k]
             if segValue == class_labels_ids[0] :
                deno[0] = deno[0] + 1
             elif segValue == class_labels_ids[1] :
                deno[1] = deno[1] + 1
             else :
                if segValue == class_labels_ids[2] :
                   deno[2] = deno[2] + 1
             if gtValue == class_labels_ids[0] :
                deno_ref[0] = deno_ref[0] + 1
                if segValue == class_labels_ids[0] :
                   nume[0] = nume[0] + 1
             elif gtValue == class_labels_ids[1] :
                deno_ref[1] = deno_ref[1] + 1
                if segValue == class_labels_ids[1] :
                   nume[1] = nume[1] + 1
             else :
                if gtValue == class_labels_ids[2] :
                   deno_ref[2] = deno_ref[2] + 1
                   if segValue == class_labels_ids[2] :
                      nume[2] = nume[2] + 1
   else :
      print "evaluation.DICE: Wrong number of classes."
   if append > 0 :
      with open(file_name, "a") as out_file:
         for i1 in range(0,nclasses):
            DICE_coef[i1] = 2.0*nume[i1]  / (deno[i1]  + deno_ref[i1])
            out_file.write('%.7f, ' %(DICE_coef[i1]))
         out_file.write('%.1f\n' %(final_time))
         out_file.write('-------------------------------------------\n')
   else :
      with open(file_name, "w") as out_file:
         for i1 in range(0,nclasses):
            DICE_coef[i1] = 2.0*nume[i1]  / (deno[i1]  + deno_ref[i1])
            out_file.write('%.7f, ' %(DICE_coef[i1]))
         out_file.write('%.1f\n' %(final_time))
         out_file.write('-------------------------------------------\n')
   
   return

#
# Function to evaluate the Accuracy.
# Input:  3D numpy integer array data with the segmentation map,
#         3D numpy integer array data_ref with the gold standard reference,
#         class labels list of strings class_labels with the string labels
#         of each brain tissue, class label integer ids class_label_ids with
#         the integer id corresponding to each class_labels element (must be
#         in the same order of appearance and valid ids are >0), number of 
#         classes nclasses (must be 3 or 4), name of the file to write the
#         results file_name, boolean flag append set to >0 if the writing on
#         the file is to be appended or 0 in other case and total segmentation
#         time final_time in seconds.
# Output: Nothing. Writes the evaluation result to the file_name file in
#         either append mode or in write mode.
@cython.boundscheck(False)
def Accuracy(np.ndarray[np.int64_t, ndim=3] data, np.ndarray[np.int64_t, ndim=3] data_ref, \
         class_labels, np.ndarray[np.int64_t, ndim=1] class_labels_ids, int nclasses, \
         str file_name, int append, double final_time) :
   cdef int i,j,k,i1,dim1,dim2,dim3,segValue,gtValue
   cdef double TP = 0
   TOTAL      = np.array(np.zeros(nclasses),dtype=np.double)
   dim1 = data.shape[0]
   dim2 = data.shape[1]
   dim3 = data.shape[2]
   if nclasses == 4 :
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
          if data_ref[i,j,k] <> 10 :
             segValue = data[i,j,k]
             gtValue  = data_ref[i,j,k]
             if gtValue == class_labels_ids[0] :
                TOTAL[0] = TOTAL[0] + 1
                if segValue == gtValue :
                   TP = TP + 1
             elif gtValue == class_labels_ids[1] :
                TOTAL[1] = TOTAL[1] + 1
                if segValue == gtValue :
                   TP = TP + 1
             elif gtValue == class_labels_ids[2] :
                TOTAL[2] = TOTAL[2] + 1
                if segValue == gtValue :
                   TP = TP + 1
             else :
                if gtValue == class_labels_ids[3] :
                   TOTAL[3] = TOTAL[3] + 1
                   if segValue == gtValue :
                      TP = TP + 1
   elif nclasses == 3 :
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
          if data_ref[i,j,k] <> 10 :
             segValue = data[i,j,k]
             gtValue  = data_ref[i,j,k]
             if gtValue == class_labels_ids[0] :
                TOTAL[0] = TOTAL[0] + 1
                if segValue == gtValue :
                   TP = TP + 1
             elif gtValue == class_labels_ids[1] :
                TOTAL[1] = TOTAL[1] + 1
                if segValue == gtValue :
                   TP = TP + 1
             else :
                if gtValue == class_labels_ids[2] :
                   TOTAL[2] = TOTAL[2] + 1
                   if segValue == gtValue :
                      TP = TP + 1
   else :
      print "evaluation.Accuracy: Wrong number of classes."
   if append > 0 :
      with open(file_name, "a") as out_file:
         out_file.write('%.7f, ' %(  (TP / sum(TOTAL))  ))
         out_file.write('%.1f\n' %(final_time))
         out_file.write('-------------------------------------------\n')
   else :
      with open(file_name, "w") as out_file:
         out_file.write('%.7f, ' %(  (TP / sum(TOTAL))  ))
         out_file.write('%.1f\n' %(final_time))
         out_file.write('-------------------------------------------\n')
   
   return

