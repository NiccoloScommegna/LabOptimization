import pycutest
import test


if __name__ == "__main__":
        
    # p = pycutest.import_problem('ALLINITU')
    
    # pycutest.print_available_sif_params('ARWHEAD')
    # p = pycutest.import_problem('ARWHEAD', sifParams={'N': 100})  # Possibili valori per N: 100, 500, 1000, 5000

    # pycutest.print_available_sif_params('BOX')
    # p = pycutest.import_problem('BOX', sifParams={'N': 10})  # Possibili valori per N: 10, 100, 1000, 10000

    # pycutest.print_available_sif_params('BOXPOWER')
    # p = pycutest.import_problem('BOXPOWER', sifParams={'N': 10})  # Possibili valori per N: 10, 100, 1000, 10000, 20000

    # p = pycutest.import_problem('BRKMCC')

    # pycutest.print_available_sif_params('BROYDN7D')
    # p = pycutest.import_problem('BROYDN7D', sifParams={'N/2': 25})  # Possibili valori per N/2: 5, 25, 50, 250, 500

    # problemi = ['ALLINITU', 'BOX', 'BOXPOWER', 'BRKMCC', 'BROYDN7D', 'DENSCHNA']


    
    methods = ['gd_armijo_with_base_function', 
               'gd_armijo_with_noisy_function', 
               'bfgs_with_base_function', 
               'bfgs_with_noisy_function', 
               'bfgs_noisy_with_noisy_function',
               'scipy_bfgs_with_base_function',
               'scipy_bfgs_with_noisy_function'
               ]
    
    # Rumore piccolo
    eps_f = 1e-8
    eps_g = 1e-8

    # Rumore grande
    # eps_f = 1e-5
    # eps_g = 1e-5

    
    # test.run_and_show_result(problem_name='ALLINITU', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000)
    # test.repeat_iterations(problem_name='ALLINITU', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True)
    # test.repeat_timing(problem_name='ALLINITU', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True)
    
    # test.run_and_show_result(problem_name='BOX', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, sif_params={'N': 10})
    # test.repeat_iterations(problem_name='BOX', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True, sif_params={'N': 10})
    # test.repeat_timing(problem_name='BOX', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True, sif_params={'N': 10})
    
    # test.run_and_show_result(problem_name='BOXPOWER', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, sif_params={'N': 10})
    # test.repeat_iterations(problem_name='BOXPOWER', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True, sif_params={'N': 10})
    # test.repeat_timing(problem_name='BOXPOWER', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True, sif_params={'N': 10})

    # test.run_and_show_result(problem_name='BRKMCC', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000)
    # test.repeat_iterations(problem_name='BRKMCC', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True)
    # test.repeat_timing(problem_name='BRKMCC', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True)

    # test.run_and_show_result(problem_name='BROYDN7D', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, sif_params={'N/2': 25})
    # test.repeat_iterations(problem_name='BROYDN7D', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True, sif_params={'N/2': 25})
    # test.repeat_timing(problem_name='BROYDN7D', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True, sif_params={'N/2': 25})

    # test.run_and_show_result(problem_name='DENSCHNA', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000)
    # test.repeat_iterations(problem_name='DENSCHNA', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True)
    # test.repeat_timing(problem_name='DENSCHNA', methods=methods, eps_f=eps_f, eps_g=eps_g, tol_factor=1, max_iter=10000, n_runs=50, quiet=True)
