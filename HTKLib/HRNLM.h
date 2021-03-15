/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*           Speech Vision and Robotics group                  */
/*           (now Machine Intelligence Laboratory)             */
/*           Cambridge University Engineering Department       */
/*           http://mi.eng.cam.ac.uk/                          */
/*                                                             */
/*           Entropic Cambridge Research Laboratory            */
/*           (now part of Microsoft)                           */
/*                                                             */
/* ----------------------------------------------------------- */
/*           Copyright: Cambridge University                   */
/*                      Engineering Department                 */
/*            2009-2015 Cambridge, Cambridgeshire UK           */
/*                      http://www.eng.cam.ac.uk               */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*     File: HRNLM.h   recurrent neural network LM handling    */
/* ----------------------------------------------------------- */

#ifndef _HRNLM_H
#define _HRNLM_H

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_STRING 100

/*-----------------------------------------------------------------------------
 *  including header file
 *-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
 *  type definition
 *-----------------------------------------------------------------------------*/
typedef double real;		/* doubles for NN weights */
/* typedef float real;		doubles for NN weights */
typedef double direct_t;

/* doubles for ME weights; TODO: check why floats are not enough for RNNME (convergence problems) */
enum FileTypeEnum {TEXT, BINARY, COMPRESSED};
/*   COMPRESSED not yet implemented */

#define  MAX_NGRAM_ORDER 20


/*Add by xc257*/
/* #define SAVEACTWITHINPUTWEIGHTMATRIX */
/* #define USEOUTPUTMATRIX */


/*-----------------------------------------------------------------------------
 *  define structure and functions related with hash table
 *-----------------------------------------------------------------------------*/

/* ----------------------- A generic hashtable  --------------- */
typedef struct _HashSlot {
    char*   key;
    void*   object;  /*  can store anything */
    struct _HashSlot* next;
} HashSlot;

typedef struct {
    unsigned int hashsize;
    unsigned int objectsize;
    HashSlot**   table;
} HashTable;

void* FindInHashTable(HashTable* hash, char* key) ;   /* given a key, return its object address  */
void  InsertToHashTable(HashTable* hash, char* key, void* obj);         /* insert a object with the name key   */
void InitalHashTable(unsigned int tablesize, unsigned int objsize, HashTable* t  );


/*  procedure to use this generic hash table:
 *  1.   InitialHashTable(MAX_TAB_SIZE, sizeof(object), hashtable)
 *          e.g., InitalHashTable(10000, sizeof(int), outVocabMap)
 *  2.   To find whether name s in hash table
 *          e.g.
 *          p=FindInHashTable(hashtable, s );
 *          if ( p==NULL ) {
 *              // not found
 *          }
 *          else {
 *              int b=*p;
 *          }
 *  3.   To insert some object with key s into hashtable:
 *          InsertToHashTable(hashtable, s,  obj)
 *          e.g.,
 *          int a= 0; string s="first object";
 *          InsertToHashTable(hashtable, s, &a);
 *          */


/*-----------------------------------------------------------------------------
 *  structure definition
 *-----------------------------------------------------------------------------*/
struct neuron {
    real ac;
    /* actual value stored in neuron */
    real er;		/*error value in neuron, used by learning algorithm */
};
struct synapse {
    real weight;	/*weight of synapse */
};
struct vocab_word {
    int cn;
    char word[MAX_STRING];

    real prob;
    int class_index;
};
typedef struct _RNNLM{

    MemHeap* x;

    /*  Prob Table */
    float *outP_arr;
    /*  input/output vocabulary  */
    int     in_num_word;            /*  input vocab size */
    int     out_num_word;           /*  output vocab size */

    HashTable in_vocab;             /*  input vocab */
    HashTable out_vocab;            /*  output vocab*/
    Boolean   setvocab;             /*  vocabulary hash table has been set ?  */
    Boolean   usefrnnlm;            /*  the structure of RNNLM is full connected ? (default: c-rnnlm)*/
    /*  vocabulary  related  */
    struct vocab_word *vocab;       /*  this records the OUTPUT vocab's word/cn/classid/  */
    int *vocab_hash;
    int vocab_hash_size;
    int fulldict_size;              /*  size of full dictionary used in ngram  */
    int in_oos_nodeid;              /*  position of oos node in in_vocab */
    int out_oos_nodeid;             /*  position of oos node in out_vocab */

    /*  model size  */
    int layer0_size;
    int layer1_size;
    int layerc_size;
    int layer2_size;
    real lognorm;

    long long direct_size;
    int direct_order;
    /*  model parameters */

    struct neuron *neu0;		/*neurons in input layer */
    struct neuron *neu1;		/*neurons in hidden layer */
    struct neuron *neuc;		/*neurons in hidden layer */
    struct neuron *neu2;		/*neurons in output layer */

    struct synapse *syn0;		/*weights between input and hidden layer */
    struct synapse *syn1;		/*weights between hidden and output layer (or hidden and compression if compression>0) */
    struct synapse *sync;		/*weights between hidden and compression layer */
    direct_t *syn_d;		/*direct parameters between input and output layer (similar to Maximum Entropy model parameters) */


    /*  class-related variables */
    int class_size;
    int **class_words;  /* class_words[0...class_size-1] ; class_words[i] holds the id-vector of words in i-th class  */
    int *class_cn;      /* class_cn[0...class_size-1]; class_cn[i] records #words in i-th class */
    int *class_max_cn;
    int old_classes;



    /*  variable not necessarily need  when using rnnlm for lmscore*/
    char train_file[MAX_STRING];
    char valid_file[MAX_STRING];
    char test_file[MAX_STRING];
    char rnnlm_file[MAX_STRING];
    char lmprob_file[MAX_STRING];

    int rand_seed;
    int debug_mode;

    int version;
    int filetype;

    int use_lmprob;
    real lambda;
    real gradient_cutoff;

    real dynamic;

    real alpha;
    real starting_alpha;
    int alpha_divide;
    double logp, llogp;
    float min_improvement;
    int iter;
    int vocab_max_size;
    int vocab_size;
    int train_words;
    int train_cur_pos;
    int counter;

    int one_iter;
    int anti_k;

    real beta;
    int history[MAX_NGRAM_ORDER];

    int bptt;
    int bptt_block;
    int *bptt_history;
    struct neuron *bptt_hidden;
    struct synapse *bptt_syn0;

    int gen;

    int independent;

        /*backup used in training: */
    struct neuron *neu0b;
    struct neuron *neu1b;
    struct neuron *neucb;
    struct neuron *neu2b;

    struct synapse *syn0b;
    struct synapse *syn1b;
    struct synapse *syncb;
    direct_t *syn_db;

    /*backup used in n-bset rescoring: */
    struct neuron *neu1b2;

    int alpha_set, train_file_set;
} RNNLM;

/*-----------------------------------------------------------------------------
 *  functions:
 *-----------------------------------------------------------------------------*/
void InitRNLM();
RNNLM* CreateRNNLM(MemHeap* x);
void LoadRNNLM(char* modelfn, RNNLM* rnnlmodel);    /* called from LoadRNLMwgt */
void LoadRNLMwgt(char* orgmodelfn, char* inmap_file, char* outmap_file,
                int in_num_word, int out_num_word, RNNLM* rnnlm);

void LoadFRNLMwgt(char* orgmodelfn, char* inmap_file, char* outmap_file,
                int in_num_word, int out_num_word, RNNLM* rnnlm);


/*  This is the function which will called from outside */
int searchRNNVocab(RNNLM* rnnlm, char *word);
void RNNLMStart(RNNLM* rnnlm) ;
void RNNLMEnd(RNNLM* rnnlm);
float RNNLMAcceptWord(RNNLM* rnnlm, int lastword, int curword);
float FRNNLMAcceptWord(RNNLM* rnnlm, int lastword, int curword);


/*-----------------------------------------------------------------------------
 *  functions used to map symbol to id
 *-----------------------------------------------------------------------------*/
int RNNLMInVocabMap(RNNLM* rnnlm, char* lab);
int RNNLMOutVocabMap(RNNLM* rnnlm, char* lab);



/*-----------------------------------------------------------------------------
 *  function used to query rnn topology
 *-----------------------------------------------------------------------------*/
Boolean isProbTableCached() ;
int GetRNNLMHiddenVectorSize(RNNLM* lm);
int GetRNNLMHistorySize(RNNLM* lm);     /*  history is used in RNNLM for direct connection */
/*-----------------------------------------------------------------------------
 *  function used to copy internal states
 *-----------------------------------------------------------------------------*/
void GetRNNLMHiddenVector(RNNLM* lm, Vector v);
/*-----------------------------------------------------------------------------
 *  function used to copy history states in input layer
 *-----------------------------------------------------------------------------*/
void GetRNNLMHistVector (RNNLM* lm, Vector v);

/*-----------------------------------------------------------------------------
 *  function used to assign history states in input layer
 *-----------------------------------------------------------------------------*/
void AssignRNNLMHistVector (RNNLM *lm, Vector v);
void copyHiddenLayerToInput(RNNLM* rnnlm);
/*-----------------------------------------------------------------------------
 *  function used to assign internal states
 *-----------------------------------------------------------------------------*/
void AssignRNNLMHiddenVector (RNNLM *lm, Vector v);

/*-----------------------------------------------------------------------------
 *  function used to copy output probabilities
 *-----------------------------------------------------------------------------*/
void GetRNNLMOutputVector (RNNLM* lm, Vector v);

/*-----------------------------------------------------------------------------
 *  function used to calculate distance between two history states (EuclidDistance used now)
  *-----------------------------------------------------------------------------*/
double CalcHistsDistance(Vector v1, Vector v2);

/*-----------------------------------------------------------------------------
 *  function used to calculate distance between two probability (cross entropy used now).
 *-----------------------------------------------------------------------------*/
float CalcProbDistance(Vector v1, Vector v2);


/*-----------------------------------------------------------------------------
 *  function used to forward in RNNLM given the history states and inputword.
 *-----------------------------------------------------------------------------*/
void RNNLMCalcProb(RNNLM* rnnlm, Vector hist, int lastword);

/*  before call this function v is a vector allocated with layer1_size
 *  dimension */
void GetRNNLMHistory(RNNLM* lm, IntVec v);
/*  before call this function v is a IntVec allocated with direct_size
 *  dimension
 *  v[1] is the most recent word!!! */
void CalOutputMatrix (Matrix outputmatrix, struct synapse *syn, int nrows, int scol, int ecol);
void UpdateMeanVariance (RNNLM* lm);
float GetVariance(int i);
void InitMeanVariance (int ndim);
void FreeMeanVariance ();

void CloneRNNLM(RNNLM* src, RNNLM* tgt);
/*  clone src rnnlm to target rnnlm
 *   -- copy src hidden vector to target
 *   -- copy history from src to target
 *   before calling this function,
 *   tgt=CreateRNNLM(MemHeap) */



/*   void sortVocab(); */

/* ----------------------- A generic hashtable  --------------- */



#ifdef __cplusplus
}
#endif


#endif
