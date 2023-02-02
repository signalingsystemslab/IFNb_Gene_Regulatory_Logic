%%
classdef model0
    % model classes
    properties
        % define the properties of the class here, (like fields of a struct)
        statename ={'0','I','N','IN'}
        beta=zeros(4,1);
        parsC=zeros(1,1); % 5 C parameters (cooperativity)
        parsT=zeros(2,1); % 6 boolean T parameters
        state=zeros(8,1);
        f;
        t;
    end
    methods
        % methods, including the constructor are defined in this block
        function obj = model0(pars) % pars length should be 11
            % class constructor
            if(nargin > 0)
                obj.parsC = pars(1);
                obj.parsT = pars(2:end);
                obj.beta = [1 1 1 obj.parsC]'; 
                obj.t = [0 obj.parsT 1]'; 
            end
        end
        
        function obj = calState2(obj,N,I)
            % calculate state matrix 2
            % only specific data points
            d = numel(N); tmp = ones(1,d);
            obj.state = [1*tmp;I;N;I.*N];
        end
        
        function obj = calState(obj,N,I)
            % calculate state matrix
            % all possible combinations of I and N (state space)
            d = numel(I); %8,N,I
            % row 1: 1
            % row 2: [irf]
            % row 3: [nfkb]
            % row 4: [irf]*[nfkb]

            obj.state = ones(4,d,d);
            for i =1:d
                obj.state(2,:,i) = I(i);
                for n = 1:d
                    obj.state(3,n,i) = N(n);
                    obj.state(4,n,i) = N(n)*I(i);
                end
            end
        end

        function obj = calF(obj)
            % calculate F value for heat map
            d=size(obj.state,2); obj.f = ones(d,d); % N ,I 
            for n = 1: d
                % rows = state
                % cols = conc. by irf
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
        
%         function obj=plotC(obj,N,I,filename)
%             contourf(log10(I),log10(N),obj.f,10);
%             colorbar;ylabel('NFkb(log10)');xlabel('IRF(log10)');
%             if nargin>3
%                 save2pdf(filename);
%             end
%         end
        
%         function obj=plotCnorm(obj,N,I,dat,filename) % linear scale
%             fscaled = obj.f/max(obj.f(:));
%             contourf(I,N,fscaled,10); % col,row order
%             colorbar;ylabel('NFkb');xlabel('IRF');
%             if nargin>3
%                 hold on;
%                 scatter(dat.irf*10,dat.nfkb*10,50,dat.ifnb,"filled");
%             end
%             if nargin>4
%                 save2pdf(filename)
%             end
%         end

        function obj=plotC(obj,N,I,dat,filename) % linear scale
            fscaled = obj.f/max(obj.f(:));
            contourf(I,N,fscaled,10); % col,row order
            colorbar;ylabel('NFkB');xlabel('IRF');
            if nargin>3
                hold on;
                scatter(dat.irf,dat.nfkb,50,dat.ifnb,"filled");
            end
            if nargin>4
                ax = gcf;
                exportgraphics(ax,filename);
            end
        end

    end
end