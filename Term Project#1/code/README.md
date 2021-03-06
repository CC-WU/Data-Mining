!! C4.5 can only run on Unix base system !!

Running C4.5 can refer to [C4.5 Tutorial](http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/tutorial.html)



The following commands were executed to produce results at the default verbosity level

    
    ./c4.5 -f [filename] > [output filename.filename extension]
    
Generate rule file
    
    ./c4.5rules -f [filename] > [output filename.filename extension]
    
    

Higher verbosity levels(1~3) may be specified to obtain statistical data calculated at runtime

    
    ./c4.5 -f [filename] -v [level] > [output filename.filename extension]
    
Generate rule file
    
    ./c4.5rules -f [filename] -v [level] > [output filename.filename extension]
    
    
Rules file has confusion matrix. For example, in the table below:

![image](https://user-images.githubusercontent.com/50508018/152553274-02b9d561-0164-4b56-957b-14aa47175006.png)

=========================================================================================

!! DataPreprocess can run on Linux & Windows !!

You can just run ./DataPreprocess/dataframe_proprocess.py to complete all data preprocessing. Some of them can be replaced, such as missing value complement, you can use mode, median, mean, etc.

./DataPreprocess Other files provide functions for plotting and analyzing data.
