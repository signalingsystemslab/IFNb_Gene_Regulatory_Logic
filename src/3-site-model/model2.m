%% model 2 
% cooperativity model
classdef model2
    % model classes
    properties
        % define the properties of the class here, (like fields of a struct)
        statename ={'0','I1','I2','N','I1I2','I1N','I2N','I1I2N'}
        beta=zeros(8,1);
        parsC=zeros(1,1); % 5 C parameters (cooperativity)
        parsT=zeros(6,1); % 6 boolean T parameters
        state=zeros(8,1);
        f;
        t;
    end
    methods
        % methods, including the constructor are defined in this block
        function obj = model2(pars) % pars length should be 11
            % class constructor
            if(nargin > 0)
                obj.parsC = pars(1);
                obj.parsT = pars(2:end);
                obj.beta = [1 1 1 1 ...  %'0','I1','I2','N',
                    obj.parsC(1) 1 1 obj.parsC(1)]';  %'I1I2','I1N','I2N','I1I2N'
                obj.t = [0 obj.parsT 1]'; 
            end
        end
        
        function obj = calState2(obj,N,I)
            % calculate state matrix 2
            d = numel(N); tmp = ones(1,d);
            obj.state = [1*tmp;I;I;N;I.^2;I.*N;I.*N;I.^2.*N];
        end
        
        function obj = calState(obj,N,I)
            % calculate state matrix
            d = numel(I); %8,N,I
            obj.state = ones(8,d,d);
            for i =1:d
                obj.state(2,:,i) = I(i);
                obj.state(3,:,i) = I(i);
                obj.state(5,:,i) = I(i)^2;
                for n = 1:d
                    obj.state(4,n,i) = N(n);
                    obj.state(6,n,i) = N(n)*I(i);
                    obj.state(7,n,i) = N(n)*I(i);
                    obj.state(8,n,i) = N(n)*I(i)^2;
                end
            end
        end

        function obj = calF(obj)
            % calculate F value for heat map
            d=size(obj.state,2); obj.f = ones(d,d); % N ,I 
            for n = 1: d
                tmps = squeeze(obj.state(:,n,:));
                obj.f(n,:) = tmps'*(obj.beta.*obj.t)...
                ./(tmps'*obj.beta);
            end
        end
        
        function obj = calF2(obj)
            % calculate F value for residue
            d=size(obj.state,2); obj.f = ones(1,d); % N ,I 
            obj.f = obj.state'*(obj.beta.*obj.t)...
                ./(obj.state'*obj.beta);

        end
        
        function obj=plotC(obj,N,I,filename)
            contourf(log10(I),log10(N),obj.f,10);
            colorbar;ylabel('NFkb(log10)');xlabel('IRF(log10)');
            if nargin>3
                save2pdf(filename);
            end
        end
        
        function obj=plotCnorm(obj,N,I,filename) % linear scale
            fscaled = obj.f/max(obj.f(:));
            contourf(I,N,fscaled,10); % col,row order
            colorbar;ylabel('NFkb');xlabel('IRF');
            if nargin>3
                save2pdf(filename)
            end
        end

    end
end