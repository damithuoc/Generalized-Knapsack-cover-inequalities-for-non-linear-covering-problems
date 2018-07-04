# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:09:15 2018

@author: Inmorales
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def EncontrarMinimo (ValAMin, IndiceMin):
    indmin=np.argmin(ValAMin)
    return (indmin)

def TruncarCovertura(itemi,partej, a, d,n,m,Trabajos):
    #esta truncando bien
    itemi-=1 #ingresan el elemento, pero trabajamos con indice
    if partej<= a[itemi]:
        return 0.
    partej-=1 #lo mismo
    llenado=np.zeros(n) #tiene cuanto ha llenado cada elemento con a
    for i in xrange(n):
        llenado[i]=sum(Trabajos[i][0:a[i]]) #como a son trabajos y esto es <a queda bien con el indice
    extra=0
    for parte in reversed(xrange(a[itemi]+1,partej+1)):#esta bien el -2? la parte anterior, en indice
        extra=extra+TruncarCovertura(itemi+1,parte,a,d,n,m,Trabajos)
    d_a=max(d-sum(llenado),0)
    if(max(d-sum(llenado)-extra,0)==0):
        return 0.
    return min(Trabajos[itemi][partej],max(d_a-extra,0.)) 



def Get_ValAMin(costos_actualizados, F,n,m,iUsados,a,d,Trabajos):
    #print(F)   
    ValAMin = []
    iValAMin =[]
    for iitem in xrange(0,n):
        for iparte in xrange(0,m):
            if (iitem,iparte) not in iUsados:
                division=1.
               # print("item,parte")
               # print(iitem+1,iparte+1)
                for iparte2 in xrange(iparte+1,m):
                    if (iparte2+1) in F[iitem][(iparte2-iparte)]:
                        division+= TruncarCovertura(iitem+1,iparte2+1,a,d,n,m,Trabajos)/Trabajos[iitem][iparte]
                ValAMin.append(costos_actualizados[iitem][iparte]/division)
                iValAMin.append((iitem,iparte))
    return (ValAMin, iValAMin)


def Resolver(Trabajos, Demanda, Costos, error):
    #los trabajos se trabajan como indices
    #Las partes no son indices, son indice +1
    n=len(Trabajos)     #cantidad de trabajos 
    m=len(Trabajos[0])  #cantidad de partes por trabajo
    SumaDual=0.              #cuanto he cubierto de la demanda
    a=np.zeros(n, dtype=np.int)       #parte hasta la que se ocupa, ojo, no el indice
    CostosActualizados=np.copy(Costos)
    ValAMin= np.copy(Costos).reshape(n*m).tolist()
    
    #print ValAMin
    F = [[[] for i in range(m)] for i in range(n)]    
    for i in range(n):
        for j in range(m):
            F[i][0].append(j+1)
    #Los indices que estan usados
    Usados=[]
    #los indices de las restricciones (indiceElemento,IndiceParte)   
    IndiceMin=[] 
    z=np.zeros((n,m))    #variables
    #contador = 0
    while (Demanda-np.sum(z))>error:
        (ValAMin,IndiceMin) = Get_ValAMin(CostosActualizados,F,n,m,Usados,a,Demanda,Trabajos)
      
        indi = EncontrarMinimo(ValAMin,IndiceMin) #entrega el i,j que se hace tight      
        Usados.append(IndiceMin[indi]) # indice del elemento que ya paso por el algoritmo       
        g = ValAMin[indi] #cuanto crece la variable dual
        SumaDual = SumaDual+g*(Demanda-np.sum(z)) #cuanto va la FO dual
        Index = IndiceMin[indi]
        for iitem in xrange(0,n): #indice de items
            for iparte in xrange(0,m): #indice de partes
                if iparte+1 in F[iitem][0]:#podria ser por cada i,j no usado
                    #print("Actualizando rest ("+str(iitem+1)+str(iparte+1)+")")                                  
                    restar = g
                    for iparte2 in xrange(iparte+1, m):
                        #print(iparte2+1,F[iitem][iparte2-iparte])
                        if iparte2+1 in F[iitem][iparte2-iparte]:
                            restar+= g*float(TruncarCovertura(iitem+1,iparte2+1,a,Demanda,n,m,Trabajos))/Trabajos[iitem][iparte]
                    CostosActualizados[iitem][iparte] -= restar
        if a[Index[0]]==Index[1]: #si el a[i] es el elemento anterior 
        # si indice es 1, el minimo es la parte 2, entonces la pregunta es si a =1
            z[Index[0]][Index[1]]=TruncarCovertura(Index[0]+1,Index[1]+1,a,Demanda,n,m,Trabajos)
            a[Index[0]]=Index[1]+1 
            
            for j in range(Index[1]+2,m+1): #el j es la parte despues del que estoy metiendo
                if j not in F[Index[0]][0] and TruncarCovertura(Index[0]+1,j,a,Demanda,n,m,Trabajos)>0:                
                    z[Index[0]][j-1]=TruncarCovertura(Index[0]+1,j,a,Demanda,n,m,Trabajos)
                    a[Index[0]]=j
                    F[Index[0]][(j-1)-Index[1]].remove(j) #la unica opcion de que llegue aca es que yo apunte al que se hizo minimo
                else: #en este caso el elemento no apoyaba
                    break
        else: #que pasa si tengo que apoyar, a cual apoyo? y como se mueven los que me apoyan?
            l=0 
            for k in xrange(Index[1],0,-1): # apoyo hacia abajo,partes anteriores j-1 a 1.                      
                # k son partes, eso va de el indice hasta 1, es decir de la parte anterior a 1.                
                if k in F[Index[0]][0]:
                    l=Index[1]+1-k       # cuantos shift son             
                    F[Index[0]][l].append(Index[1]+1)
                    break
            # veo como se mueven los de adelante
            for k in xrange(Index[1]+2,m+1): # las partes de adelante
                if k in F[Index[0]][(k-1)-Index[1]]: # si apunta al que ingresa
                    F[Index[0]][(k-1)-Index[1]].remove(k)
                    F[Index[0]][(k-1)-Index[1]+l].append(k)
        F[Index[0]][0].remove(Index[1]+1) # elimino el j por si entro o por si lo movi.
        
        # Usados.append((Index[0],Index[1]+1)) #se marca como "usado" que paso por el algoritmo
        # calcular el conjunto a minimizar y sacar el >0 del minimo
        ValAMin=[]        
        IndiceMin=[]
   #tengo que los a entregan la solución
    
    print ("Dual=",SumaDual,"<=","Primal=" ,np.sum(z*Costos))
    print ("2Dual",2*SumaDual,">=","Primal=", np.sum(z*Costos))
    print ("Gap = ",np.round(np.sum(z*Costos)/SumaDual,8))
    return z, np.round(np.sum(z*Costos)/(SumaDual),8)



def Crear_Funciones(n,m,eps,plot):
    #cuantas generadoras grandes, pequeñas y medianas
    peq = int(np.ceil(float(n)/21*6))
    gra = int(np.floor(float(n)/21*6))    
    med = n-peq-gra    
    #promedio de "pendiente" de las generadoras 
    a_peq = 0.009
    v_peq = 0.0015    
    a_med = 0.0065
    v_med = 0.0015
    a_gra = 0.005
    v_gra =0.0015
    a=[]    
    a = np.random.uniform(a_peq-v_peq,a_peq+v_peq,peq)
    a = np.append(a,np.random.uniform(a_med-v_med,a_med+v_med,med))
    a = np.append(a,np.random.uniform(a_gra-v_gra,a_gra+v_gra,gra))
    
    #factor para sacar la produccion minima
    aux = np.random.randint(25,70,n).astype(float)/100*np.random.randint(10,15,n).astype(float)/100
    #primedio de produccion maxima de cada generadora segun tamaño
    g_peq = 35.05
    d_peq = 24.95
    g_med = 212.478
    d_med = 86.8
    g_gra = 604.958
    d_gra = 191.64
    G_max = np.random.randint(g_peq - d_peq, g_peq + d_peq, peq)
    G_max = np.append(G_max,np.random.randint(g_med - d_med, g_med + d_med, med))
    G_max = np.append(G_max,np.random.randint(g_gra - d_gra, g_gra+ d_gra, gra))
    
    G_min = (G_max*aux).astype(int)+1
    b=-G_min*2*a
    
    c_peq = 10
    c_med = 20
    c_gra = 30
    
    c = []
    c = (G_min*G_min*a+b*G_min + c_peq)[0:peq]
    c = np.append(c,(G_min*G_min*a+b*G_min + c_med)[peq:(med+peq)])
    c = np.append(c,(G_min*G_min*a+b*G_min + c_gra)[(med+peq):n])    
    
    if plot==1:
        #plot
        fig= plt.figure()    
        axes=fig.add_subplot(111)        
        for i in range(0,n):
            ejex=[]
            ejey=[]
            for x in range(G_min[i],G_max[i],1):
                y=x**2*a[i]+b[i]*x +c[i]
                ejex.append(x)
                ejey.append(y)
            if i<peq:
                axes.plot(ejex,ejey, 'r')
            elif i<peq+med:
                axes.plot(ejex,ejey, 'g')
            else:
                axes.plot(ejex,ejey, 'b')
        #plt.xlim( (0, 150) )
        #plt.ylim( (0, 150) )
        plt.show()
    
    Trabajos= []
    Costos = []
    Trabajos.append([eps]*n)
    Costos.append((a*G_min.astype(float)**2 + b*G_min.astype(float) +c)/eps )
    Trabajos.append(G_min-eps)    
    Costos.append([0]*n)
    aux_x=G_min.astype(float)
    delta_x = (G_max-G_min).astype(float)/(m-1)
    for parte in range(0,m-1):
        Trabajos.append(delta_x)
        delta_y = a*((aux_x+delta_x)**2-aux_x**2)+b*(delta_x)
        Costos.append(delta_y/delta_x)
        aux_x += delta_x
    Trabajos = np.array(Trabajos).T
    Costos = np.array(Costos).T
    return (Costos,Trabajos,a,b,c,G_min,G_max)





def graficar(Trabajos,Costos,n,a,b,c,G_min,G_max,xmax,ymax):
    #plot
    peq = int(np.ceil(float(n)/21*6))
    gra = int(np.floor(float(n)/21*6))    
    med = n-peq-gra 
    fig= plt.figure()    
    axes=fig.add_subplot(111)        
    for i in range(0,n):
        ejex=[]
        ejey=[]
        for x in range(G_min[i],G_max[i],1):
            y=x**2*a[i]+b[i]*x +c[i]
            ejex.append(x)
            ejey.append(y)
        if i<peq:
            axes.plot(ejex,ejey, 'r')
        elif i<peq+med:
            axes.plot(ejex,ejey, 'g')
        else:
            axes.plot(ejex,ejey, 'b')
    plt.xlim( (0, xmax) )
    plt.ylim( (0, ymax) )
    plt.show()

def simular_sing(plot, escenario, eps, m,escenario2 = 0,title =""):
    pos = np.random.get_state()[2]-1
    seed=np.random.get_state()[1][pos]
    #print(seed)
    np.random.seed(seed)
    G_max = [802.53, 780.6, 584.22, 558.2, 549.72, 532.46, 380., 275.3, 
             181.75, 177.54, 177., 103.68, 77.27, 44.6, 28.64, 14.32, 
             6.8, 6.4, 6.06, 3., 2.68, 2.]
    color = ['r', 'r', 'b','b','b','b','b', 'g','g','g','g','g','g','g','g',
             'g','g','g','g','g','g','g',]
    xLabels = ['U', 'CC', 'CTM', 'ANG', 'CCH', 'CC_KELAR', 'CC_SALTA', 'NTO', 'TAR', 
       'CTH', 'CTA', 'SUTA', 'TG', 'UG', 'MIMB', 'GMAR', 'INACAL', 'ZOFRIEST', 
       'ZOFRI', 'TECNET_1_6', 'CUMMINS', 'AGB']
    G_max.reverse()
    color.reverse()
    xLabels.reverse()
    
    N= len(G_max)
    med = 5
    gra = 2    
    peq = N-med-gra    
    x=range(1,N+1)

    a_peq = 0.009
    v_peq = 0.0015    
    a_med = 0.0065
    v_med = 0.0015
    a_gra = 0.005
    v_gra = 0.0015
    a=[]    
    a = np.random.uniform(a_peq-v_peq,a_peq+v_peq,peq)
    a = np.append(a,np.random.uniform(a_med-v_med,a_med+v_med,med))
    a = np.append(a,np.random.uniform(a_gra-v_gra,a_gra+v_gra,gra))
    # factor para sacar la produccion minima
    aux = np.random.randint(25,70,N).astype(float)/100*np.random.randint(10,15,N).astype(float)/100
    G_min = (G_max*aux).astype(int)+1
    b = -G_min*2*a

    pond_cf=[] #pond_cf[0] peq, pond_cf[1] peq, pond_cf[2] peq    
    if escenario==1: # costos mas grande, mas costo, con baja relevancia
        pond_cf=[]        
        pond_cf.append(0.2)
        pond_cf.append(0.2)
        pond_cf.append(0.2)
    elif escenario==2: # mas grande, mas costo inicial con alta relevancia
        pond_cf=[]
        pond_cf.append(0.5) 
        pond_cf.append(0.5)
        pond_cf.append(0.5)
    elif escenario==3: #costo inicial parecido
        pond_cf=[]
        pond_cf.append(0.5)
        pond_cf.append(0.2)
        pond_cf.append(0.2)
    g_max = np.array(G_max)
    cf_peq = ((G_min*(1-pond_cf[0])+pond_cf[0]*g_max)*(G_min*(1-pond_cf[0])+pond_cf[0]*g_max)*a+(G_min*(1-pond_cf[0])+pond_cf[0]*g_max)*b)[0:peq]
    cf_med = ((G_min*(1-pond_cf[1])+pond_cf[1]*g_max)*(G_min*(1-pond_cf[1])+pond_cf[1]*g_max)*a+(G_min*(1-pond_cf[1])+pond_cf[1]*g_max)*b)[peq:(med+peq)]
    cf_gra = ((G_min*(1-pond_cf[2])+pond_cf[2]*g_max)*(G_min*(1-pond_cf[2])+pond_cf[2]*g_max)*a+(G_min*(1-pond_cf[2])+pond_cf[2]*g_max)*b)[(med+peq):N]   
   
    c = -2*(G_min*G_min*a+b*G_min)
    c[0:peq] = c[0:peq] + cf_peq
    c[peq:(med+peq)] = c[peq:(med+peq)] + cf_med
    c[(med+peq):N] = c[(med+peq):N] + cf_gra
    if escenario2 == 1:
        c = c + max(c)*10
    
    if plot ==1:
       # y_mini = max(G_min*G_min*a+b*G_min + c)
       # x_mini = max(G_min)
        
        fig= plt.figure()    
        axes=fig.add_subplot(111)        
        for i in range(0,N):
            ejex=[]
            ejey=[]
            for x in range(G_min[i],int(G_max[i]),1):
                y=x**2*a[i]+b[i]*x +c[i]
                ejex.append(x)
                ejey.append(y)
            if i<peq:
                axes.plot(ejex,ejey, 'r')
            elif i<peq+med:
                axes.plot(ejex,ejey, 'g')
            else:
                axes.plot(ejex,ejey, 'b')
        #plt.xlim( (0, x_mini+20) )
        #plt.ylim( (0, y_mini+20 ) )
        plt.title("Semilla = "+str(seed))
        plt.xlabel("Cubrimiento")
        plt.ylabel("Costo")    
        plt.savefig("./"+title+"_seed"+str(seed)+".png")        
        plt.show()  
    
    
    # Me falta crear la funcion discretizada para retornar.
    Trabajos= []
    Costos = []
    Trabajos.append([eps]*N)
    Costos.append((a*G_min.astype(float)**2 + b*G_min.astype(float) +c)/eps )
    Trabajos.append(G_min-eps)    
    Costos.append([0]*N)
    aux_x=G_min.astype(float)
    delta_x = (G_max-G_min).astype(float)/(m-1)
    #para ver el epsilon error de aproximación de las funciones    
    Error = []
    Error2 = []
    for parte in range(0,m-1):
        Trabajos.append(delta_x)
        delta_y = a*((aux_x+delta_x)**2-aux_x**2)+b*(delta_x)
        fxi0 = (a*(aux_x**2)+b*aux_x+c).copy()
        Costos.append(delta_y/delta_x)
        x_aux = ( Costos[-1]-b)/(2*a)
        Error2.append((fxi0 + Costos[-1]*(x_aux - aux_x) -(a*(x_aux**2)+b*x_aux+c))/(a*(x_aux**2)+b*x_aux+c))
        #print("bien la derivada?")
        #print(x_aux-aux_x)
        aux_x += delta_x
        fxi1 = (a*(aux_x**2)+b*aux_x+c).copy()
        Error.append((fxi1-fxi0)/fxi0)
    Trabajos = np.array(Trabajos).T
    Costos = np.array(Costos).T
    error = np.array(Error2).T.max()
    return (Costos,Trabajos,a,b,c,G_min,G_max,error)    



# desde acá parte el main
alguna=True

opciones =[("escenario iii.1)",3,1,1./4),("escenario iii.2)",3,1,1./2),("escenario iii.3)",3,1,3./4),
("escenario i.1)",3,0,1./4),("escenario i.2)",3,0,1./2),("escenario i.3)",3,0,3./4),
("escenario ii.1)",1,0,1./4),("escenario ii.2)",1,0,1./2),("escenario ii.3)",1,0,3./4),
("escenario iv.1)",2,0,1./4),("escenario iv.2)",2,0,1./2),("escenario iv.3)",2,0,3./4)
]
Resultados =[("escenario","media_Gap", "max_Gap", "min_Gap", "std_Gap",
              "media_error", "max_error", "min_error", "std_error",
              "media_tiempo", "max_tiempo", "min_tiempo", "std_tiempo",
              "media_m", "max_m", "min_m", "std_m")]

for (title,escenario,escenario2,ctedem) in opciones:
    Gap=[]
    errores =[]
    Tiempo = []
    m_list = []
    for i in range(100):
        print(title, i )
        eps= 0.00001
        #(Costos,Trabajos,a,b,c,G_min,G_max) = Crear_Funciones(20,10,eps,1)
        plot = 1*(i<2)
        #escenario = 1
        errori = 0.6

        m=2
        while errori>0.05:
            m+=1            
            (Costos,Trabajos,a,b,c,G_min,G_max,error) = simular_sing(plot,escenario,eps,m,escenario2,title)
            errori = error
            #print(error)
        #print(m)
        m_list.append(m)
        errores.append(error)     #el error es el epsilon
        Demanda = np.sum(Trabajos)*ctedem 
        n=len(Trabajos)
        m=len(Trabajos[0])
        start_time = time.time()
        (z,gap) = Resolver(Trabajos,Demanda,Costos, error)
        end_time = time.time()
        Tiempo.append(end_time - start_time)
        print("--- %s seconds ---" % (end_time - start_time))
        Gap.append(gap)    
        f = a*sum(z.T)**2+b*sum(z.T)+c
        suma = 0
        for i in range(0,len(f)):
            if sum(z.T)[i]>0:
                suma += f[i]
        print(sum(z.T))
    Resultados.append((title,
        np.mean(Gap),np.max(Gap),np.min(Gap), np.std(Gap),
        np.mean(errores), np.max(errores), np.min(errores), np.std(errores),
        np.mean(Tiempo), np.max(Tiempo), np.min(Tiempo), np.std(Tiempo),
        np.mean(m_list), np.max(m_list), np.min(m_list), np.std(m_list)  
        
    ))
#f=open("../Version1/ejemplos/Resultados_vfinal.txt","w")
f=open("./Resultados_vfinal.txt","w")
print >> f,str(Resultados)
f.close()   
        
