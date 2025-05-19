import re
import numpy as np
import sys

# Variáveis globais para armazenar o estado do problema
caminho_arquivo = ''
eh_max = True
num_variaveis = 0
num_restricoes = 0
coef_objetivo = []
restricoes_esq = []
restricoes_dir = []
tipos_restricoes = []

# Matrizes do Simplex Revisado
A = None                # Matriz de coeficientes
b = None                # Termos independentes
c = None                # Coeficientes objetivo
vars_basicas = []       # Variáveis básicas
vars_nao_basicas = []   # Variáveis não básicas
B = None                # Matriz de base
B_inv = None            # Inversa da base
x_B = None              # Solução básica
y = None                # Variáveis duais

# Variáveis artificiais
tem_artificial = False
vars_artificiais = []
num_artificiais = 0
num_folga_excesso = 0
precos_sombra = []

# Contadores de iterações
num_iteracoes = 0
num_iteracoes_fase1 = 0

def ler_arquivo():
    global eh_max, num_variaveis, coef_objetivo, restricoes_esq, restricoes_dir, tipos_restricoes, num_restricoes
    
    with open(caminho_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
    
    # Função objetivo
    linha_obj = linhas[0].strip()
    if 'Max' in linha_obj:
        eh_max = True
    elif 'Min' in linha_obj:
        eh_max = False
    
    # Coeficientes da função objetivo
    expr_obj = linha_obj.split('=')[1].strip()
    termos = re.findall(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)', expr_obj)
    
    indices_var = [int(idx) for _, idx in termos]
    num_variaveis = max(indices_var)
    coef_objetivo = np.zeros(num_variaveis)
    
    for coef, var_idx in termos:
        coef = coef.replace(' ', '')
        if coef == '+' or coef == '':
            coef = 1
        elif coef == '-':
            coef = -1
        else:
            coef = float(coef)
        coef_objetivo[int(var_idx) - 1] = float(coef)
    
    # Restrições
    for i in range(1, len(linhas)):
        linha = linhas[i].strip()
        if not linha:
            continue
        
        if '<=' in linha:
            esq, dir = linha.split('<=')
            tipo = '<='
        elif '>=' in linha:
            esq, dir = linha.split('>=')
            tipo = '>='
        elif '=' in linha and not ('>' in linha or '<' in linha):
            esq, dir = linha.split('=')
            tipo = '='
        else:
            continue
        
        # Lado esquerdo
        termos = re.findall(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)', esq)
        coefs_esq = np.zeros(num_variaveis)
        
        for coef, var_idx in termos:
            coef = coef.replace(' ', '')
            if coef == '+' or coef == '':
                coef = 1
            elif coef == '-':
                coef = -1
            else:
                coef = float(coef)
            coefs_esq[int(var_idx) - 1] = float(coef)
        
        # Lado direito
        val_dir = float(dir.strip())
        
        restricoes_esq.append(coefs_esq)
        restricoes_dir.append(val_dir)
        tipos_restricoes.append(tipo)
    
    num_restricoes = len(restricoes_dir)
    print("Número de variáveis originais:", num_variaveis)
    print("Número de restrições originais:", num_restricoes)

def forma_padrao():
    global A, b, c, tem_artificial, vars_artificiais, num_artificiais, num_folga_excesso
    global vars_basicas, vars_nao_basicas, B, B_inv
    
    # Contagem de variáveis extras
    num_folga_excesso = num_restricoes
    num_artificiais = 0
    
    for tipo in tipos_restricoes:
        if tipo == '=' or tipo == '>=':
            num_artificiais += 1
            tem_artificial = True
    
    # Total de variáveis
    total_vars = num_variaveis + num_folga_excesso + num_artificiais
    
    print("Número de variáveis de folga/excesso:", num_folga_excesso)
    print("Número de variáveis artificiais:", num_artificiais)
    print("Total de variáveis:", total_vars)

    # Inicializar matrizes
    A = np.zeros((num_restricoes, total_vars))
    b = np.array(restricoes_dir)
    c = np.zeros(total_vars)
    
    # Função objetivo
    for j in range(num_variaveis):
        if eh_max:
            c[j] = coef_objetivo[j]
        else:
            c[j] = -coef_objetivo[j]
    
    # Matriz de restrições com variáveis auxiliares
    indice_artificial = num_variaveis + num_folga_excesso
    vars_artificiais = []
    
    for i in range(num_restricoes):
        tipo = tipos_restricoes[i]
        
        # Coeficientes originais
        for j in range(num_variaveis):
            A[i, j] = restricoes_esq[i][j]
        
        # Variáveis de folga/excesso/artificiais
        if tipo == '<=':
            A[i, num_variaveis + i] = 1  # Folga
        elif tipo == '>=':
            A[i, num_variaveis + i] = -1  # Excesso
            A[i, indice_artificial] = 1  # Artificial
            vars_artificiais.append(indice_artificial)
            indice_artificial += 1
        elif tipo == '=':
            A[i, indice_artificial] = 1  # Artificial
            vars_artificiais.append(indice_artificial)
            indice_artificial += 1
    
    # Correção para RHS negativo
    for i in range(num_restricoes):
        if b[i] < 0:
            b[i] = -b[i]
            A[i, :] = -A[i, :]
    
    # Inicializar base
    vars_basicas = []
    for i in range(num_restricoes):
        if tipos_restricoes[i] == '<=':
            vars_basicas.append(num_variaveis + i)
        else:
            indice = vars_artificiais.pop(0)
            vars_basicas.append(indice)
    
    # Variáveis não-básicas
    vars_nao_basicas = [j for j in range(total_vars) if j not in vars_basicas]
    
    # Matriz de base e sua inversa
    B = A[:, vars_basicas]
    B_inv = np.linalg.inv(B)
    
    print("\nMatriz A (coeficientes):")
    print(A)
    print("\nMatriz B (base):")
    print(B)
    print("\nVetor c (coeficientes da função objetivo):")
    print(c)

def simplex_revisado():
    global x_B, y, vars_basicas, vars_nao_basicas, B, B_inv, num_iteracoes, precos_sombra
    
    # Solução básica inicial
    x_B = B_inv @ b
    
    # Reiniciar contador de iterações
    iteracoes = 0
    
    for iteracao in range(100):  # Limite de iterações
        iteracoes += 1
        
        # Passo 1: Variáveis duais
        c_B = c[vars_basicas]
        y = c_B @ B_inv
        
        # Passo 2: Custos reduzidos
        var_entrada = -1
        custo_max = 1e-10
        
        for k, j in enumerate(vars_nao_basicas):
            a_j = A[:, j]
            custo_reduzido = c[j] - y @ a_j
            
            if custo_reduzido > custo_max:
                custo_max = custo_reduzido
                var_entrada = j
        
        # Verificar otimalidade
        if custo_max <= 1e-10:
            break
        
        # Passo 3: Direção simplex
        a_q = A[:, var_entrada]
        d = B_inv @ a_q
        
        # Verificar ilimitabilidade
        if all(d_i <= 1e-10 for d_i in d):
            print("Problema ilimitado.")
            return False
        
        # Passo 4: Teste da razão
        theta = float('inf')
        indice_saida = -1
        
        for i in range(len(x_B)):
            if d[i] > 1e-10:
                razao = x_B[i] / d[i]
                if razao < theta:
                    theta = razao
                    indice_saida = i       
        var_saida = vars_basicas[indice_saida]
        
        # Passo 5: Atualizar solução
        x_B = x_B - theta * d
        x_B[indice_saida] = theta
        
        # Passo 6: Atualizar inversa
        E = np.eye(len(vars_basicas))
        E[:, indice_saida] = -d / d[indice_saida]
        E[indice_saida, indice_saida] = 1 / d[indice_saida]
        B_inv = E @ B_inv
        
        # Passo 7: Atualizar conjuntos de variáveis
        vars_basicas[indice_saida] = var_entrada
        vars_nao_basicas.remove(var_entrada)
        vars_nao_basicas.append(var_saida)
        B = A[:, vars_basicas]
    
    # Armazenar número de iterações
    num_iteracoes = iteracoes
    
    # Preços sombra
    precos_sombra = y
    if not eh_max:
        precos_sombra = -y
    
    return True

def fase_um():
    global c, num_iteracoes_fase1, num_iteracoes, vars_basicas, vars_nao_basicas, B, B_inv, x_B
    
    if not tem_artificial:
        num_iteracoes_fase1 = 0
        return True
    
    # Salvar vetor de custo original
    c_original = c.copy()
    
    # Função objetivo da Fase I
    c = np.zeros_like(c)
    for j in range(c.shape[0]):
        if j >= num_variaveis + num_folga_excesso:
            c[j] = -1  # Minimizar artificiais
    
    # Resolver problema auxiliar
    if not simplex_revisado():
        print("Fase I falhou: problema sem solução viável.")
        return False
    
    # Registrar iterações da fase I
    num_iteracoes_fase1 = num_iteracoes
    num_iteracoes = 0  # Zerar para contar a fase II separadamente
    
    # Verificar variáveis artificiais
    for i, var in enumerate(vars_basicas):
        if var >= num_variaveis + num_folga_excesso and x_B[i] > 1e-6:
            print("Problema sem solução viável.")
            return False
    
    # Remover artificiais da base
    for i in range(len(vars_basicas)):
        if vars_basicas[i] >= num_variaveis + num_folga_excesso:
            for j in vars_nao_basicas:
                if j < num_variaveis + num_folga_excesso:
                    a_j = A[:, j]
                    d = B_inv @ a_j
                    
                    if abs(d[i]) > 1e-6:
                        # Troca de base
                        theta = x_B[i] / d[i]
                        x_B = x_B - theta * d
                        x_B[i] = theta
                        
                        # Atualizar inversa
                        E = np.eye(len(vars_basicas))
                        E[:, i] = -d / d[i]
                        E[i, i] = 1 / d[i]
                        B_inv = E @ B_inv
                        
                        # Atualizar variáveis
                        var_saida = vars_basicas[i]
                        vars_basicas[i] = j
                        vars_nao_basicas.remove(j)
                        vars_nao_basicas.append(var_saida)
                        break
    
    # Restaurar função objetivo original
    c = c_original
    B = A[:, vars_basicas]
    
    return True

def solucao_primal():
    solucao = np.zeros(num_variaveis)
    
    for i, var in enumerate(vars_basicas):
        if var < num_variaveis:
            solucao[var] = x_B[i]
    
    # Valor da função objetivo
    valor_obj = 0
    for i, var in enumerate(vars_basicas):
        if var < num_variaveis:
            if eh_max:
                valor_obj += coef_objetivo[var] * x_B[i]
            else:
                valor_obj -= coef_objetivo[var] * x_B[i]
    
    return solucao, valor_obj

def solucao_dual():
    return precos_sombra

def imprimir_problema_primal():
    print("\n==== PROBLEMA PRIMAL ====")
    
    # Função objetivo
    objetivo = "Maximizar" if eh_max else "Minimizar"
    func_obj = f"{objetivo} Z = "
    
    termos = []
    for i, coef in enumerate(coef_objetivo):
        if coef != 0:
            if coef == 1:
                termos.append(f"x{i+1}")
            elif coef == -1:
                termos.append(f"-x{i+1}")
            else:
                termos.append(f"{coef}x{i+1}")
    
    func_obj += " + ".join(termos).replace("+ -", "- ")
    print(func_obj)
    
    # Restrições
    print("Sujeito a:")
    for i in range(num_restricoes):
        restricao = ""
        termos = []
        
        for j, coef in enumerate(restricoes_esq[i]):
            if coef != 0:
                if coef == 1:
                    termos.append(f"x{j+1}")
                elif coef == -1:
                    termos.append(f"-x{j+1}")
                else:
                    termos.append(f"{coef}x{j+1}")
        
        restricao += " + ".join(termos).replace("+ -", "- ")
        restricao += f" {tipos_restricoes[i]} {restricoes_dir[i]}"
        print(restricao)
    
    # Não-negatividade
    print("x_j >= 0 para todo j")

def imprimir_problema_dual():
    print("\n==== PROBLEMA DUAL ====")
    
    # O tipo do problema dual é o oposto do primal
    objetivo_dual = "Minimizar" if eh_max else "Maximizar"
    
    # Função objetivo do dual
    func_obj_dual = f"{objetivo_dual} W = "
    termos_dual = []
    
    for i, b_i in enumerate(restricoes_dir):
        if b_i != 0:
            if b_i == 1:
                termos_dual.append(f"y{i+1}")
            elif b_i == -1:
                termos_dual.append(f"-y{i+1}")
            else:
                termos_dual.append(f"{b_i}y{i+1}")
    
    func_obj_dual += " + ".join(termos_dual).replace("+ -", "- ")
    print(func_obj_dual)
    
    # Restrições do dual
    print("Sujeito a:")
    
    for j in range(num_variaveis):
        restricao_dual = ""
        termos_dual = []
        
        for i in range(num_restricoes):
            coef = restricoes_esq[i][j]
            if coef != 0:
                if coef == 1:
                    termos_dual.append(f"y{i+1}")
                elif coef == -1:
                    termos_dual.append(f"-y{i+1}")
                else:
                    termos_dual.append(f"{coef}y{i+1}")
        
        restricao_dual += " + ".join(termos_dual).replace("+ -", "- ")
        
        # Tipo da restrição dual depende do problema primal
        if eh_max:
            # Primal max => Dual min
            restricao_dual += f" >= {coef_objetivo[j]}"
        else:
            # Primal min => Dual max
            restricao_dual += f" <= {coef_objetivo[j]}"
        
        print(restricao_dual)

# Código principal direto (sem definir uma função main)
if len(sys.argv) < 2:
    print("Uso: python simplex_revisado.py arquivo_entrada.txt")
else:
    caminho_arquivo = sys.argv[1]
    
    # Executar o Simplex Revisado
    ler_arquivo()
    imprimir_problema_primal()
    imprimir_problema_dual()
    forma_padrao()
    
    if fase_um():
        if simplex_revisado():
            # Soluções
            solucao_primal_val, valor_obj = solucao_primal()
            solucao_dual_val = solucao_dual()
            
            # Resultados
            print("\n==== SOLUÇÃO PRIMAL ====")
            print(f"Valor Ótimo: {valor_obj}")
            print("Variáveis:")
            for i, val in enumerate(solucao_primal_val):
                print(f"  x{i+1} = {val}")
            
            print("\n==== SOLUÇÃO DUAL (PREÇOS SOMBRA) ====")
            for i, val in enumerate(solucao_dual_val):
                # Verificar o sinal de acordo com a tabela
                sinal_ajustado = val
                tipo_restricao = tipos_restricoes[i]
                
                if eh_max:
                    # Para problema primal de maximização
                    if tipo_restricao == ">=":
                        sinal_ajustado = -val  # Inverter sinal para restrição >=
                    
                else:
                    # Para problema primal de minimização
                    if tipo_restricao == "<=":
                        sinal_ajustado = -val  # Inverter sinal para restrição <=
                
                print(f"  y{i+1} = {sinal_ajustado}")
            
            # Estatísticas
            print("\n==== ESTATÍSTICAS ====")
            print(f"Número de iterações na Fase I: {num_iteracoes_fase1}")
            print(f"Número de iterações na Fase II: {num_iteracoes}")
            print(f"Total de iterações: {num_iteracoes_fase1 + num_iteracoes}")
        else:
            print("O problema é ilimitado.")
    else:
        print("O problema não tem solução viável.")