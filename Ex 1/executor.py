from poly import GenPolyData
from poly import get_data
from poly import Regressor
from poly import plt
import shutil

def test_data_generator(number,g,generated):


    # GENERATE TEST DATA
    if not generated:
        g.genera(1000, 'data/' + str(number) + '/dati.p5.std01.te')
    else:
        shutil.copy('data/' + str(number-1) + '/dati.p5.std01.te','data/' + str(number) + '/dati.p5.std01.te')

    return True


def training_data_generator(number,g):

    data_tests = ((number + 1) * 10)

    # GENERATE TRAIN DATA
    print "data test: "+str(data_tests)

    g.genera(data_tests, 'data/'+str(number)+'/dati.p5.std01.tr')



#Test Function
def tester(number):
    Xtr,Ytr = get_data('data/'+str(number)+'/dati.p5.std01.tr')
    Xte,Yte = get_data('data/'+str(number)+'/dati.p5.std01.te')

    tr_mse = []
    te_mse = []

    for p in range(1,21):
        R = Regressor(p,0.01)
        tr_mse.append(R.train(Xtr,Ytr,'test/'+str(number)+'/fig_train.%02d.png'%p))
        te_mse.append(R.test(Xte,Yte,'test/'+str(number)+'/fig_test.%02d.png'%p))

    plt.clf()
    plt.plot(range(1,21), tr_mse,'b')
    plt.plot(range(1,21), te_mse,'r')
    plt.xlabel("degree")
    plt.ylabel("mse")
    plt.savefig("test/"+str(number)+"/mse.png",format='png')



def run():

    print "Start Regression testing...."

    generated = False
    g = GenPolyData(10, 0.25)

    for x in range(0,5):

        print "Generating data for test: " + str(x)
        generated =  test_data_generator(x,g,generated)

        print "Generating training data for test: " + str(x)
        training_data_generator(x,g)

        print "Running test: " + str(x)
        tester(x)

    print "End Regression testing...."


run()