%% 
function quadriga_lib_mex_tests

single_run = false;

% Qudriga-lib path
tmp = which('quadriga_lib.version');
if isempty(strfind(tmp,'+quadriga_lib/version.mex'))
    current_dir = pwd;
    cd('../');
    current_dir2 = pwd;
    addpath(current_dir2)
    cd(current_dir);
end 

% MOxUnit Setup
tmp = which('MOxUnitTestSuite');
if isempty(tmp)
    current_dir = pwd;
    cd('../external/MOxUnit-master/MOxUnit');
    moxunit_set_path();
    cd(current_dir);
end

% List of all available tests
tests = dir('mex_tests/test*.m');
addpath( [pwd,'/mex_tests']);
N = numel( tests );

clc
disp(['Running ',num2str(N),' tests'])
if single_run % Run tests one-by-one
    for n = 1 : N
        subFunctionName = tests(n).name(1:end-2);
        disp(subFunctionName)
        test_suite=MOxUnitTestSuite();
        test_case = MOxUnitFunctionHandleTestCase(subFunctionName,'run_mex_tests', str2func( subFunctionName ));
        test_suite=addTest(test_suite, test_case);
        disp(run(test_suite));
    end
else % Run all tests
    test_suite=MOxUnitTestSuite();
    for n = 1 : N
        subFunctionName = tests(n).name(1:end-2);
        test_case = MOxUnitFunctionHandleTestCase(subFunctionName,...
            'run_mex_tests', str2func( subFunctionName ));
        test_suite=addTest(test_suite, test_case);
    end
    tic
    disp(run(test_suite));
    toc
end




