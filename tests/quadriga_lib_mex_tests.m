% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function quadriga_lib_mex_tests

single_run = false;

% quadriga-lib path
tmp = which('quadriga_lib.version');
if isempty(strfind(tmp,'+quadriga_lib/version.mex'))
    current_dir = pwd;
    cd('../');
    current_dir2 = pwd;
    addpath(current_dir2)
    cd(current_dir);
end 

% MOxUnit path
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
all_passed = true;
if single_run % Run tests one-by-one
    for n = 1 : N
        subFunctionName = tests(n).name(1:end-2);
        disp(subFunctionName)
        test_suite=MOxUnitTestSuite();
        test_case = MOxUnitFunctionHandleTestCase(subFunctionName,'run_mex_tests', str2func( subFunctionName ));
        test_suite=addTest(test_suite, test_case);
        test_report = run(test_suite);
        disp(test_report);
        all_passed = all_passed && wasSuccessful(test_report);
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
    test_report = run(test_suite);
    disp(test_report);
    toc
    all_passed = wasSuccessful(test_report);
end

% End Octave or MATLAB with an error, so that "make test" and CI runs stop on a failed test
if ~all_passed
    error('quadriga_lib_mex_tests: some tests failed');
end
