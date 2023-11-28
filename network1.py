import numpy as np

class ReseauNeural(object):

    def __init__(self, tailles):
        self.num_layers = len(tailles)
        self.tailles = tailles
        self.biais = [np.random.randn(y, 1) for y in tailles[1:]]
        self.poids = [np.random.randn(y, x) for x, y in zip(tailles[:-1], tailles[1:])]

    def propagation_directe(self, a,f):
        for b, w in zip(self.biais, self.poids):
            a = f(np.dot(w, a)+b)
        return a
    
    def Descente_gradient(self, donnees_entrainement, epochs, eta,f,f_prime,
            donnees_test=None):
        L=[]
        n_test = len(donnees_test)
        n = len(donnees_entrainement)
        for j in range(1,epochs+1):

            for x,y in donnees_entrainement:
                self.mettre_a_jour((x,y), eta,f,f_prime)
            if donnees_test:
                print(f"epoque {j}: {self.evaluation(donnees_test,f)} / {n_test}")
                L.append([j,self.evaluation(donnees_test,f)/ n_test*100])
            else:
                print("Epoch {0} est completé".format(j))
            
        return L


    def mettre_a_jour(self, L, eta,f,f_prime):
        nabla_b = [np.zeros(b.shape) for b in self.biais]
        nabla_w = [np.zeros(w.shape) for w in self.poids]

        delta_nabla_b, delta_nabla_w = self.retroprop(L[0], L[1],f,f_prime)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.poids = [w-(eta)*nw for w, nw in zip(self.poids, nabla_w)]
        self.biais = [b-(eta)*nb for b, nb in zip(self.biais, nabla_b)]

    def retroprop(self, x, y,f,f_prime):
        nabla_b = [np.zeros(b.shape) for b in self.biais]
        nabla_w = [np.zeros(w.shape) for w in self.poids]
        activation = x
        activations = [x] # liste pour stocker toutes les activations, couche par couche
        zs = [] # liste pour stocker tous les vecteurs z, couche par couche
        #passage en avant (calculer les activations correspondant à x)
        for b, w in zip(self.biais, self.poids):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = f(z)
            activations.append(activation)
        # passage en arrière
        delta = self.derivee_cout(activations[-1], y) * f_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = f_prime(z)
            delta = np.dot(self.poids[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluation(self, donnees_test,f):
        resultat_test = [(np.argmax(self.propagation_directe(x,f)), y)
                        for (x, y) in donnees_test]
        return sum(int(x == y) for (x, y) in resultat_test)

    def derivee_cout(self, activation_sortie, y):
        return (activation_sortie-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
def tanh(z):
    return (np.tanh(z)+1)/2
def tanh_prime(z):
    return (1-tanh(z)**2)/2
def arctan(z):
    return np.arctan(z)/np.pi+0.5
def arctan_prime(z):
    return (1/(1+z**2))/(np.pi)